# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Anytime-valid sequential test for early-stopping benchmark evaluation.

This module implements sequential decision rules based on classical building
blocks for deciding, as samples stream in, whether a model's
accuracy sits **above**, **below**, or **within** a target band
``[target - margin, target + margin]``:

* **SPRT likelihood screen** – a fixed-dataset comparison between two edge
  populations: ``k0 = N * p_lo`` and ``k1 = N * p_hi`` successes (rounded to
  integer case counts). It asks whether the observed data looks much closer to
  one edge than the other; it is not a confidence-interval certificate that the
  true accuracy is outside the band.

* **Bayesian finite-population screen** – a Beta prior over the full dataset
  success rate induces a Beta-binomial prior over the total number of successful
  items ``K`` in the fixed population. After each sample, the posterior is
  updated with the same without-replacement likelihood family used by SPRT.

* **aGRAPA** – the *approximate Growth Rate Adaptive to the Particular
  Alternative* betting strategy of Waudby-Smith & Ramdas, *"Estimating means of
  bounded random variables by betting"*, in its **sampling-without-replacement
  (WoR)** form: the dataset is a fixed finite population of ``N`` items, so the
  null "population mean is ``m``" pins the conditional mean of the next draw to
  ``mu_t(m) = (N*m - S_{t-1}) / (N - t + 1)`` given the running sum ``S_{t-1}``.
  A test martingale (capital process)
  ``K_t(m) = prod_i (1 + lambda_i (X_i - mu_i(m)))`` is maintained for a grid of
  candidate means ``m``; the bet ``lambda_i`` is set adaptively from running
  estimates of the mean and variance. Inverting the family of martingales
  yields an **anytime-valid confidence sequence** for the true accuracy that is
  valid under continuous monitoring (i.e. you may peek after every sample).
  Candidates whose conditional mean leaves ``[0, 1]`` are logically impossible
  (the already-observed sum contradicts them) and are rejected outright.

Because both objects are likelihood/wealth-ratio processes that are valid under
optional stopping, they can be monitored after every observation without the
error-rate inflation that would plague repeatedly applying a fixed-n test.

The outcomes ``X_t`` are assumed bounded in ``[0, 1]`` (e.g. per-sample binary
accuracy, or a pass-rate fraction).
"""

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Dict, List, Literal, Optional

import numpy as np

Strategy = Literal['sprt', 'agrappa', 'fixed', 'bayes']
Decision = Literal['above', 'below', 'within', 'undecided']
RiskAssessment = Literal['conservative', 'balance', 'balanced', 'aggressive']
RISK_PROFILES = {
    'conservative': {'type1': 0.05, 'type2': 0.10},
    'balance': {'type1': 0.05, 'type2': 0.20},
    'balanced': {'type1': 0.05, 'type2': 0.20},
    'aggressive': {'type1': 0.10, 'type2': 0.20},
}

# Numerical guards keeping probabilities strictly inside (0, 1).
_EPS = 1e-6
# Shrinkage applied to the betting fraction so the wealth process stays strictly
# positive for any outcome in [0, 1] (aGRAPA-style truncation).
_BET_SHRINK = 0.5
# Floor on the running variance estimate used to size bets.
_VAR_FLOOR = 1e-4


@dataclass
class StoppingDecision:
    """Snapshot of the sequential test after the latest observation."""

    decision: Decision
    """Geometric decision relative to the zone: ``'above'`` / ``'below'`` /
    ``'within'`` / ``'undecided'``. Framing-neutral; see :attr:`verdict` for the
    go/no-go reading."""

    decided_by: Optional[str]
    """Which engine produced the decision: ``'sprt'``, ``'agrappa'``, ``'fixed'``, ``'bayes'`` or ``None``."""

    n: int
    """Number of observations consumed so far."""

    mean: float
    """Running point estimate of the accuracy (plain sample mean)."""

    ci_lower: float
    """Lower end of the anytime-valid confidence sequence."""

    ci_upper: float
    """Upper end of the anytime-valid confidence sequence."""

    target_range: List[float]
    """The ``[p_lo, p_hi]`` acceptable range around the target (``target ±
    margin``); technically the indifference zone / region of practical
    equivalence."""

    target: float
    """The target accuracy bar the verdict is measured against."""

    sprt_llr: float
    """Current SPRT log-likelihood ratio (``H1: p_hi`` over ``H0: p_lo``)."""

    sprt_decision: Decision
    """Standalone SPRT verdict (never ``'within'``)."""

    cs_decision: Decision
    """Standalone confidence-sequence (aGRAPA) verdict."""

    extra: Dict = field(default_factory=dict)
    """Diagnostics (evidence cutoffs, thresholds, config echo)."""

    @property
    def decided(self) -> bool:
        return self.decision != 'undecided'

    @property
    def decision_explanation(self) -> str:
        """Plain-English explanation of what the deciding engine did."""
        if self.decided_by == 'sprt':
            if self.decision == 'above':
                direction = 'upper'
                opposite = 'lower'
            elif self.decision == 'below':
                direction = 'lower'
                opposite = 'upper'
            else:
                direction = 'one'
                opposite = 'the other'
            return (
                f'Observed data is more consistent with the {direction} edge hypothesis than the '
                f'{opposite} edge hypothesis, enough to pass the configured evidence cutoff. '
                'This is not a confidence-interval certificate that accuracy is outside the band.'
            )
        if self.decided_by == 'agrappa':
            return 'Decision is certified by the anytime-valid confidence sequence.'
        if self.decided_by == 'bayes':
            return (
                'Decision is based on posterior mass under a finite-population Bayesian model with a Beta prior. '
                'The likelihood is sampling-without-replacement over the fixed dataset.'
            )
        if self.decided_by == 'fixed':
            return 'Decision is based on a single Wilson confidence interval at the fixed sample budget.'
        if self.decided_by == 'budget':
            return 'Decision is based on the point estimate after the configured sample budget was reached.'
        return 'No early decision has been made.'

    @property
    def verdict(self) -> str:
        """Plain-language call against the target bar — always resolves (a finite
        benchmark always yields a score), three-way around the ±margin zone:

        * ``'PASS'`` — clears the bar (above the zone).
        * ``'FAIL'`` — misses the bar (below the zone).
        * ``'BORDERLINE'`` — sits inside the ±margin zone, i.e. at the bar within
          tolerance. *Not* a failure — just close enough that the exact side
          doesn't matter.

        For a confident early call this comes from :attr:`decision`; otherwise it
        falls back to the point estimate vs. the zone edges. Once all samples have
        run that estimate is exact, so the call is definitive — early stopping only
        governs *how many samples were needed*, never whether we can score.
        """
        if self.decision == 'above':
            return 'PASS'
        if self.decision == 'below':
            return 'FAIL'
        if self.decision == 'within':
            return 'BORDERLINE'
        lo, hi = self.target_range  # undecided -> point estimate vs the zone
        if self.mean > hi:
            return 'PASS'
        if self.mean < lo:
            return 'FAIL'
        return 'BORDERLINE'

    def to_dict(self) -> Dict:
        p_lo, p_hi = self.target_range
        return {
            'verdict': self.verdict,
            'decision_explanation': self.decision_explanation,
            'decided_early': self.decided,
            'decision': self.decision,
            'decided_by': self.decided_by,
            'n': self.n,
            'mean': self.mean,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'target': self.target,
            'target_range': self.target_range,
            'grey_zone': {
                'lower': p_lo,
                'upper': p_hi,
                'center': self.target,
                'lower_margin': self.target - p_lo,
                'upper_margin': p_hi - self.target,
            },
            'p_lo': p_lo,
            'p_hi': p_hi,
            'p0': p_lo,
            'p1': p_hi,
            'sprt_llr': self.sprt_llr,
            'sprt_decision': self.sprt_decision,
            'cs_decision': self.cs_decision,
            **self.extra,
        }


class SequentialStopper:
    """
    Online SPRT/aGRAPA/fixed/Bayesian test for a bounded-mean target band.

    Feed observations one at a time via :meth:`update`; query :meth:`result`
    (or its cached :attr:`last_result`) to see whether the accuracy has been
    resolved against the band. The test only reports ``decided`` once at least
    ``min_samples`` observations have been seen.

    Args:
        target: Centre of the target accuracy band, in ``(0, 1)``.
        margin: Half-width of the band; the band is ``[target-margin, target+margin]``.
        alpha: Type-I error level. The confidence sequence has coverage
            ``1 - alpha``.
        beta: Type-II error level used in the SPRT evidence cutoffs.
        strategy: Which engine drives the stop:

            * ``'sprt'`` (default) — SPRT likelihood screen only. It reports
              directional edge-likelihood calls and never certifies ``'within'``.
            * ``'agrappa'`` — **band-exclusion estimation**, CS only. ``'above'``/
              ``'below'`` certify the true mean is outside the band edge; use this
              to answer "where is the accuracy?" with a strict guarantee.
            * ``'fixed'`` — no sequential peeking; run to ``max_samples`` and
              make one Wilson-interval decision against ``target``.
            * ``'bayes'`` — Bayesian finite-population posterior. Directional
              calls require posterior mass outside the grey-zone edge.
        min_samples: Minimum observations before a decision may be reported.
        total_samples: Total number of samples in the dataset being sampled from.
        max_samples: Optional budget cap. If reached without a confident call,
            the test stops anyway and returns a ``BORDERLINE``/point-estimate
            verdict (``decided_by='budget'``) — i.e. "we won't spend more samples
            chasing precision in the don't-care zone." ``None`` runs unbounded.
        grid_size: Number of candidate means used to invert the betting
            martingale into a confidence sequence.
    """

    def __init__(
        self,
        target: float,
        margin: float,
        *,
        alpha: float = 0.05,
        beta: float = 0.20,
        strategy: Strategy = 'sprt',
        min_samples: int = 30,
        total_samples: int,
        max_samples: Optional[int] = None,
        grid_size: int = 999,
        bayes_prior_alpha: float = 1.0,
        bayes_prior_beta: float = 1.0,
        judge_false_negative_rate: float = 0.0,
        judge_false_positive_rate: float = 0.0,
    ):
        self.target = float(target)
        self.margin = float(margin)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.strategy: Strategy = self.normalize_strategy(strategy)
        self.min_samples = max(1, int(min_samples))
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.total_samples = int(total_samples)
        self.bayes_prior_alpha = float(bayes_prior_alpha)
        self.bayes_prior_beta = float(bayes_prior_beta)
        self.judge_false_negative_rate = float(judge_false_negative_rate)
        self.judge_false_positive_rate = float(judge_false_positive_rate)
        self._judge_scale = 1.0 - self.judge_false_negative_rate - self.judge_false_positive_rate
        if not (0.0 <= self.judge_false_negative_rate < 1.0):
            raise ValueError(f'judge_false_negative_rate must be in [0, 1), got {judge_false_negative_rate}')
        if not (0.0 <= self.judge_false_positive_rate < 1.0):
            raise ValueError(f'judge_false_positive_rate must be in [0, 1), got {judge_false_positive_rate}')
        if self._judge_scale <= 0.0:
            raise ValueError(
                'judge_false_negative_rate + judge_false_positive_rate must be < 1. '
                f'Got {judge_false_negative_rate} + {judge_false_positive_rate}.'
            )
        self.p_lo, self.p_hi = self.validate_config(
            target=self.target, margin=self.margin, alpha=self.alpha, beta=self.beta, strategy=self.strategy
        )
        self.q_target = self._observed_from_true(self.target)
        self.q_lo = self._observed_from_true(self.p_lo)
        self.q_hi = self._observed_from_true(self.p_hi)
        if self.strategy == 'fixed':
            self.q_lo = self.q_target
            self.q_hi = self.q_target
        if self.total_samples <= 0:
            raise ValueError(f'total_samples must be > 0, got {total_samples}')
        if self.bayes_prior_alpha <= 0.0 or self.bayes_prior_beta <= 0.0:
            raise ValueError(
                f'bayes_prior_alpha and bayes_prior_beta must be > 0, got '
                f'{bayes_prior_alpha}, {bayes_prior_beta}'
            )

        self._k0 = max(0, min(self.total_samples, math.floor(self.total_samples * self.q_lo)))
        self._k1 = max(0, min(self.total_samples, math.ceil(self.total_samples * self.q_hi)))
        if self._k0 >= self._k1 and self.strategy != 'fixed':
            raise ValueError(
                f'total_samples={self.total_samples} is too small for distinct SPRT hypotheses: '
                f'k0={self._k0}, k1={self._k1}'
            )

        # --- aGRAPA confidence-sequence state -------------------------------
        self._grid = np.linspace(_EPS, 1.0 - _EPS, int(grid_size))
        self._log_wealth = np.zeros_like(self._grid)
        self._log_threshold = math.log(1.0 / self.alpha)

        # --- running moments (predictable bets use pre-observation stats) ---
        self._n = 0
        self._sum = 0.0
        self._sum_sq = 0.0

        # --- SPRT likelihood-screen state -----------------------------------
        self._llr = 0.0
        self._sprt_upper = math.log((1.0 - self.beta) / self.alpha)
        self._sprt_lower = math.log(self.beta / (1.0 - self.alpha))

        # --- Bayesian finite-population state -------------------------------
        self._population_success_counts = None
        self._log_factorial = None
        self._bayes_log_prior = None
        if self.strategy == 'bayes':
            self._population_success_counts = np.arange(self.total_samples + 1, dtype=int)
            self._log_factorial = np.array([math.lgamma(i + 1.0) for i in range(self.total_samples + 1)])
            self._bayes_log_prior = self._build_bayes_log_prior()
        self._bayes_posterior_cache: Optional[Dict] = None

        self.last_result: Optional[StoppingDecision] = None

    def _observed_from_true(self, true_rate: float) -> float:
        """Map true accuracy ``p`` to expected judge-positive rate ``q``."""
        return self.judge_false_positive_rate + self._judge_scale * true_rate

    def _true_from_observed(self, observed_rate: float) -> float:
        """Map observed judge-positive rate ``q`` back to latent true accuracy ``p``."""
        true_rate = (observed_rate - self.judge_false_positive_rate) / self._judge_scale
        return max(0.0, min(1.0, true_rate))

    def _true_interval_from_observed(self, lower: float, upper: float) -> tuple:
        """Convert an observed-rate interval to corrected true-accuracy space."""
        return self._true_from_observed(lower), self._true_from_observed(upper)

    @staticmethod
    def normalize_strategy(strategy: str) -> Strategy:
        """Normalize supported strategy aliases to their report-facing value."""
        key = str(strategy).lower()
        if key in ('bayesian', 'baysian'):
            return 'bayes'
        return key

    @staticmethod
    def validate_config(
        target: float,
        margin: float,
        alpha: float = 0.05,
        beta: float = 0.20,
        strategy: Strategy = 'sprt',
    ) -> tuple:
        """Validate shared stopper settings and return the clipped target range."""
        strategy = SequentialStopper.normalize_strategy(strategy)
        if not (0.0 < target < 1.0):
            raise ValueError(f'target must be in (0, 1), got {target}')
        if margin <= 0.0 and strategy != 'fixed':
            raise ValueError(f'margin must be > 0, got {margin}')
        if not (0.0 < alpha < 1.0):
            raise ValueError(f'alpha must be in (0, 1), got {alpha}')
        if not (0.0 < beta < 1.0):
            raise ValueError(f'beta must be in (0, 1), got {beta}')
        if strategy not in ('sprt', 'agrappa', 'fixed', 'bayes'):
            raise ValueError(f"strategy must be one of 'sprt'|'agrappa'|'fixed'|'bayes', got {strategy!r}")
        if strategy == 'fixed':
            return target, target
        return max(_EPS, target - margin), min(1.0 - _EPS, target + margin)

    @staticmethod
    def risk_levels(risk_assessment: RiskAssessment) -> tuple:
        """Return ``(alpha, beta)`` for a named risk-assessment profile."""
        key = str(risk_assessment).lower()
        if key not in RISK_PROFILES:
            raise ValueError(
                f"risk_assessment must be one of {sorted(RISK_PROFILES.keys())}, got {risk_assessment!r}"
            )
        profile = RISK_PROFILES[key]
        return profile['type1'], profile['type2']

    @staticmethod
    def estimate_margin(target: float, total_samples: int, sample_budget: int, alpha: float, beta: float) -> float:
        """Estimate the target-band half-width implied by a fixed sample budget.

        Uses the normal approximation for detecting a proportion difference with
        type-I risk ``alpha`` and type-II risk ``beta`` under sampling without
        replacement from a finite dataset.
        """
        SequentialStopper.validate_config(target=target, margin=_EPS, alpha=alpha, beta=beta)
        total = int(total_samples)
        budget = int(sample_budget)
        if total <= 0:
            raise ValueError(f'total_samples must be > 0, got {total_samples}')
        if not (1 <= budget <= total):
            raise ValueError(f'sample_budget must be in [1, total_samples], got {sample_budget}')

        if total == 1 or budget >= total:
            margin = _EPS
        else:
            z_type1 = NormalDist().inv_cdf(1.0 - alpha)
            z_type2 = NormalDist().inv_cdf(1.0 - beta)
            finite_population_correction = (total - budget) / (total - 1)
            variance = target * (1.0 - target) * finite_population_correction / budget
            margin = (z_type1 + z_type2) * math.sqrt(max(variance, 0.0))

        max_margin = min(target - _EPS, 1.0 - target - _EPS)
        return max(_EPS, min(max_margin, margin))

    # ------------------------------------------------------------------ #
    # Update                                                              #
    # ------------------------------------------------------------------ #
    def update(self, x: float) -> StoppingDecision:
        """Consume one observation ``x`` in ``[0, 1]`` and return the new result."""
        x = float(x)
        if not (0.0 <= x <= 1.0):
            raise ValueError(f'observation must be in [0, 1], got {x}')
        if self._n >= self.total_samples:
            raise ValueError(f'cannot consume more than total_samples={self.total_samples} observations')

        if self.strategy in ('sprt', 'agrappa'):
            # Predictable bets: size them from statistics over *previous* samples
            # only, which keeps the wealth process a valid test martingale.
            mu_prev = (0.5 + self._sum) / (self._n + 1)
            # Variance around the running mean, with a Beta(1/2, 1/2)-style prior.
            var_prev = (0.25 + self._sum_sq - 2 * mu_prev * self._sum + self._n * mu_prev ** 2) / (
                self._n + 1
            )
            var_prev = max(var_prev, _VAR_FLOOR)

            # Sampling without replacement from the finite population of N items:
            # under the null "population mean is m", the next draw has conditional
            # mean mu_t(m) = (N*m - S_{t-1}) / (N - t + 1). Candidates with
            # mu_t(m) outside [0, 1] are logically impossible given the observed
            # sum (each item is in [0, 1]) and stay impossible, so reject them.
            remaining = self.total_samples - self._n
            mu_null = (self.total_samples * self._grid - self._sum) / remaining
            feasible = (mu_null >= 0.0) & (mu_null <= 1.0)
            mu_null = np.clip(mu_null, _EPS, 1.0 - _EPS)

            # GRAPA bet per grid point, truncated so 1 + lambda*(x - mu_t(m)) > 0
            # for any x in [0, 1] (worst cases x=0 and x=1), with extra shrinkage.
            lam = (mu_prev - mu_null) / var_prev
            upper = _BET_SHRINK / mu_null
            lower = -_BET_SHRINK / (1.0 - mu_null)
            lam = np.clip(lam, lower, upper)
            self._log_wealth = np.where(
                feasible, self._log_wealth + np.log1p(lam * (x - mu_null)), math.inf
            )

        # Commit moments.
        self._n += 1
        self._sum += x
        self._sum_sq += x * x
        if self.strategy != 'fixed':
            self._update_sprt_llr()
        self._bayes_posterior_cache = None

        self.last_result = self._build_result()
        return self.last_result

    def _build_bayes_log_prior(self) -> np.ndarray:
        """Return log Beta-binomial prior mass over total success count K."""
        if self._population_success_counts is None or self._log_factorial is None:
            raise RuntimeError('Bayesian prior requested before Bayesian state was initialized.')
        k = self._population_success_counts.astype(float)
        n = float(self.total_samples)
        a = self.bayes_prior_alpha
        b = self.bayes_prior_beta
        log_comb = self._log_factorial[self.total_samples] - self._log_factorial[self._population_success_counts]
        log_comb = log_comb - self._log_factorial[self.total_samples - self._population_success_counts]
        log_beta_num = np.array([math.lgamma(float(ki) + a) + math.lgamma(n - float(ki) + b) for ki in k])
        log_beta_den = math.lgamma(n + a + b)
        log_prior_den = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        return log_comb + log_beta_num - log_beta_den - log_prior_den

    def _update_sprt_llr(self) -> None:
        """Update the SPRT log-likelihood ratio.

        The null/alternative are fixed-count populations: ``H0`` has ``k0``
        successes in ``N`` items and ``H1`` has ``k1`` successes. After observing
        ``s`` successes and ``f`` failures, the common hypergeometric denominator
        cancels, leaving ``log C(k1,s)C(N-k1,f) - log C(k0,s)C(N-k0,f)``.
        """
        s = self._sum
        f = self._n - self._sum
        ll1 = self._finite_population_log_likelihood(self._k1, s, f)
        ll0 = self._finite_population_log_likelihood(self._k0, s, f)
        if ll1 == -math.inf and ll0 == -math.inf:
            self._llr = 0.0
        elif ll1 == -math.inf:
            self._llr = -math.inf
        elif ll0 == -math.inf:
            self._llr = math.inf
        else:
            self._llr = ll1 - ll0

    def _finite_population_log_likelihood(self, success_cases: int, successes: float, failures: float) -> float:
        """Return log P(observed successes/failures | fixed success_cases), up to a constant."""
        total = self.total_samples
        failure_cases = total - success_cases
        if successes > success_cases + _EPS or failures > failure_cases + _EPS:
            return -math.inf
        return self._log_falling(success_cases, successes) + self._log_falling(failure_cases, failures)

    def _finite_population_log_likelihood_grid(self, successes: float, failures: float) -> np.ndarray:
        """Vectorized log likelihood over all finite-population success counts K."""
        if self._population_success_counts is None or self._log_factorial is None:
            raise RuntimeError('Bayesian likelihood requested before Bayesian state was initialized.')
        total = self.total_samples
        k = self._population_success_counts
        failure_cases = total - k
        feasible = (successes <= k + _EPS) & (failures <= failure_cases + _EPS)
        out = np.full(total + 1, -math.inf, dtype=float)
        if not feasible.any():
            return out

        success_int = int(round(successes))
        failure_int = int(round(failures))
        if abs(successes - success_int) <= _EPS and abs(failures - failure_int) <= _EPS:
            feasible_k = k[feasible]
            out[feasible] = (
                self._log_factorial[feasible_k]
                - self._log_factorial[feasible_k - success_int]
                + self._log_factorial[failure_cases[feasible]]
                - self._log_factorial[failure_cases[feasible] - failure_int]
            )
            return out

        out[feasible] = np.array([
            self._log_falling(int(ki), successes) + self._log_falling(int(total - ki), failures)
            for ki in k[feasible]
        ])
        return out

    @staticmethod
    def _log_falling(population: int, draws: float) -> float:
        """Log falling factorial ``population * ... * (population - draws + 1)``."""
        if draws <= 0.0:
            return 0.0
        return math.lgamma(population + 1.0) - math.lgamma(population - draws + 1.0)

    # ------------------------------------------------------------------ #
    # Decision logic                                                      #
    # ------------------------------------------------------------------ #
    def _confidence_sequence(self) -> tuple:
        """Return ``(ci_lower, ci_upper)`` of the current confidence sequence."""
        in_ci = self._log_wealth < self._log_threshold
        if not in_ci.any():
            # Fully rejected everywhere (rare, tiny grid); fall back to point.
            mean = self._sum / self._n if self._n else 0.5
            return mean, mean
        idx = np.flatnonzero(in_ci)
        # The true mean can sit between grid points, so round outward one step
        # to stay conservative. This matters at the WoR feasibility cutoff,
        # where wealth jumps to +inf between adjacent grid points.
        lo = self._grid[max(idx[0] - 1, 0)]
        hi = self._grid[min(idx[-1] + 1, len(self._grid) - 1)]
        return float(lo), float(hi)

    def _wilson_interval(self) -> tuple:
        """Return the fixed-sample Wilson score interval for the current mean."""
        if self._n == 0:
            return 0.0, 1.0
        p_hat = self._sum / self._n
        z = NormalDist().inv_cdf(1.0 - self.alpha / 2.0)
        z2 = z * z
        denom = 1.0 + z2 / self._n
        center = (p_hat + z2 / (2.0 * self._n)) / denom
        half_width = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * self._n)) / self._n) / denom
        return max(0.0, center - half_width), min(1.0, center + half_width)

    def _bayes_posterior_summary(self) -> Dict:
        """Return posterior probabilities and credible interval for the finite-population Bayesian model."""
        if self._bayes_posterior_cache is not None:
            return self._bayes_posterior_cache
        if self._population_success_counts is None or self._bayes_log_prior is None:
            raise RuntimeError('Bayesian posterior requested before Bayesian state was initialized.')

        log_likelihood = self._finite_population_log_likelihood_grid(self._sum, self._n - self._sum)
        log_posterior = self._bayes_log_prior + log_likelihood
        finite = np.isfinite(log_posterior)
        if not finite.any():
            posterior = np.zeros_like(log_posterior)
            posterior[int(round(self._sum))] = 1.0
        else:
            max_log = float(np.max(log_posterior[finite]))
            posterior = np.zeros_like(log_posterior)
            posterior[finite] = np.exp(log_posterior[finite] - max_log)
            posterior_sum = float(np.sum(posterior))
            posterior = posterior / posterior_sum if posterior_sum > 0.0 else posterior

        rates = self._population_success_counts / self.total_samples
        prob_below = float(np.sum(posterior[rates < self.q_lo]))
        prob_above = float(np.sum(posterior[rates > self.q_hi]))
        prob_within = max(0.0, 1.0 - prob_below - prob_above)
        cdf = np.cumsum(posterior)
        lower_q = self.alpha / 2.0
        upper_q = 1.0 - self.alpha / 2.0
        lower_idx = int(np.searchsorted(cdf, lower_q, side='left'))
        upper_idx = int(np.searchsorted(cdf, upper_q, side='left'))
        lower_idx = min(max(lower_idx, 0), self.total_samples)
        upper_idx = min(max(upper_idx, 0), self.total_samples)
        summary = {
            'posterior_below': prob_below,
            'posterior_within': prob_within,
            'posterior_above': prob_above,
            'ci_lower': lower_idx / self.total_samples,
            'ci_upper': upper_idx / self.total_samples,
            'success_ci': [lower_idx, upper_idx],
        }
        self._bayes_posterior_cache = summary
        return summary

    def _cs_decision(self, ci_lower: float, ci_upper: float) -> Decision:
        if ci_lower > self.q_hi:
            return 'above'
        if ci_upper < self.q_lo:
            return 'below'
        if ci_lower >= self.q_lo and ci_upper <= self.q_hi:
            return 'within'
        return 'undecided'

    def _sprt_decision(self) -> Decision:
        if self._llr >= self._sprt_upper:
            return 'above'  # observed data is closer to the upper edge hypothesis
        if self._llr <= self._sprt_lower:
            return 'below'  # observed data is closer to the lower edge hypothesis
        return 'undecided'

    def _bayes_decision(self, posterior: Dict) -> Decision:
        if posterior['posterior_above'] >= 1.0 - self.alpha:
            return 'above'
        if posterior['posterior_below'] >= 1.0 - self.alpha:
            return 'below'
        if posterior['posterior_within'] >= 1.0 - self.beta:
            return 'within'
        return 'undecided'

    def _build_result(self) -> StoppingDecision:
        if self.strategy == 'fixed':
            observed_ci_lower, observed_ci_upper = self._wilson_interval()
            cs_dec = 'undecided'
            sprt_dec = 'undecided'
            bayes = None
        elif self.strategy == 'bayes':
            bayes = self._bayes_posterior_summary()
            observed_ci_lower, observed_ci_upper = bayes['ci_lower'], bayes['ci_upper']
            cs_dec = 'undecided'
            sprt_dec = self._sprt_decision()
        else:
            bayes = None
            observed_ci_lower, observed_ci_upper = self._confidence_sequence()
            cs_dec = self._cs_decision(observed_ci_lower, observed_ci_upper)
            sprt_dec = self._sprt_decision()
        ci_lower, ci_upper = self._true_interval_from_observed(observed_ci_lower, observed_ci_upper)
        observed_mean = self._sum / self._n if self._n else 0.0
        corrected_mean = self._true_from_observed(observed_mean) if self._n else 0.0

        decision: Decision = 'undecided'
        decided_by: Optional[str] = None
        if self.strategy == 'fixed':
            budget = self.max_samples if self.max_samples is not None else self.total_samples
            if self._n >= min(budget, self.total_samples):
                if observed_ci_lower > self.q_target:
                    decision = 'above'
                elif observed_ci_upper < self.q_target:
                    decision = 'below'
                else:
                    decision = 'within'
                decided_by = 'fixed'
        elif self.strategy == 'agrappa':
            # Band-exclusion *estimation*: only the anytime-valid CS decides, so
            # 'above'/'below' certify the true mean is outside the band edge, and
            # 'within' certifies it inside. Use this to answer "where is the
            # accuracy?" with a strict guarantee.
            decision, decided_by = cs_dec, ('agrappa' if cs_dec != 'undecided' else None)
        elif self.strategy == 'sprt':
            decision, decided_by = sprt_dec, ('sprt' if sprt_dec != 'undecided' else None)
        elif self.strategy == 'bayes':
            decision = self._bayes_decision(bayes)
            decided_by = 'bayes' if decision != 'undecided' else None

        # Suppress a premature stop below the minimum sample floor.
        if self._n < self.min_samples:
            decision, decided_by = 'undecided', None

        # Budget cap: out of samples without a confident call -> stop anyway with
        # the point-estimate call (BORDERLINE if inside the zone). We accept the
        # lower precision rather than keep sampling in the don't-care region.
        elif decision == 'undecided' and self.max_samples is not None and self._n >= self.max_samples:
            mean = self._sum / self._n
            decision = 'above' if mean > self.q_hi else ('below' if mean < self.q_lo else 'within')
            decided_by = 'budget'

        extra = {
            'strategy': self.strategy,
            'alpha': self.alpha,
            'beta': self.beta,
            'margin': self.margin,
            'min_samples': self.min_samples,
            'total_samples': self.total_samples,
            'sprt_success_hypotheses': [self._k0, self._k1],
            'sprt_h0_success_rate': self._k0 / self.total_samples,
            'sprt_h1_success_rate': self._k1 / self.total_samples,
            'observed_mean': observed_mean,
            'observed_ci_lower': observed_ci_lower,
            'observed_ci_upper': observed_ci_upper,
            'observed_target': self.q_target,
            'observed_target_range': [self.q_lo, self.q_hi],
            'q0': self.q_lo,
            'q1': self.q_hi,
            'judge_error_correction': {
                'false_negative_rate': self.judge_false_negative_rate,
                'false_positive_rate': self.judge_false_positive_rate,
                'scale': self._judge_scale,
                'enabled': (
                    self.judge_false_negative_rate > 0.0
                    or self.judge_false_positive_rate > 0.0
                ),
                'formula': 'q = fpr + (1 - fnr - fpr) * p',
            },
            'sprt_evidence_cutoffs': [self._sprt_lower, self._sprt_upper],
            'sprt_boundaries': [self._sprt_lower, self._sprt_upper],
            'log_wealth_threshold': self._log_threshold,
            'interval_type': (
                'wilson' if self.strategy == 'fixed'
                else ('bayesian_credible_interval' if self.strategy == 'bayes' else 'confidence_sequence')
            ),
        }
        if bayes is not None:
            extra.update({
                'bayes_prior': {
                    'distribution': 'beta',
                    'alpha': self.bayes_prior_alpha,
                    'beta': self.bayes_prior_beta,
                },
                'bayes_posterior_below': bayes['posterior_below'],
                'bayes_posterior_within': bayes['posterior_within'],
                'bayes_posterior_above': bayes['posterior_above'],
                'bayes_success_credible_interval': bayes['success_ci'],
                'bayes_decision_thresholds': {
                    'above': 1.0 - self.alpha,
                    'below': 1.0 - self.alpha,
                    'within': 1.0 - self.beta,
                },
            })

        return StoppingDecision(
            decision=decision,
            decided_by=decided_by,
            n=self._n,
            mean=corrected_mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            target_range=[self.p_lo, self.p_hi],
            target=self.target,
            sprt_llr=self._llr,
            sprt_decision=sprt_dec,
            cs_decision=cs_dec,
            extra=extra,
        )

    def result(self) -> StoppingDecision:
        """Return the latest result, building an empty one if no data yet."""
        if self.last_result is None:
            ci_lower, ci_upper = self._true_interval_from_observed(float(self._grid[0]), float(self._grid[-1]))
            return StoppingDecision(
                decision='undecided', decided_by=None, n=0, mean=0.0,
                ci_lower=ci_lower, ci_upper=ci_upper,
                target_range=[self.p_lo, self.p_hi], target=self.target, sprt_llr=0.0,
                sprt_decision='undecided', cs_decision='undecided',
            )
        return self.last_result
