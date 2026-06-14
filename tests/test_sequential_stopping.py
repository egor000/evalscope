# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the sequential early-stopping test."""

import math
import random
import unittest
from collections import Counter

from evalscope.metrics.sequential_stopping import SequentialStopper


def _run(true_p, *, target=0.6, margin=0.08, strategy='sprt', alpha=0.05,
         min_samples=30, max_samples=None, n=20000, seed=0):
    """Stream Bernoulli(true_p) outcomes until the test decides or data runs out."""
    rng = random.Random(seed)
    stopper = SequentialStopper(
        target=target, margin=margin, strategy=strategy, alpha=alpha,
        min_samples=min_samples, max_samples=max_samples, total_samples=n,
    )
    for _ in range(n):
        res = stopper.update(1.0 if rng.random() < true_p else 0.0)
        if res.decided:
            return res
    return stopper.result()


class TestSequentialStopper(unittest.TestCase):

    def test_validation(self):
        with self.assertRaises(ValueError):
            SequentialStopper(target=1.5, margin=0.05, total_samples=100)
        with self.assertRaises(ValueError):
            SequentialStopper(target=0.5, margin=0.0, total_samples=100)
        with self.assertRaises(ValueError):
            SequentialStopper(target=0.5, margin=0.05, alpha=0.0, total_samples=100)
        with self.assertRaises(ValueError):
            SequentialStopper(target=0.5, margin=0.05, strategy='nope', total_samples=100)
        with self.assertRaises(TypeError):
            SequentialStopper(target=0.5, margin=0.05)

    def test_band_computation(self):
        t = SequentialStopper(target=0.6, margin=0.08, total_samples=100)
        self.assertAlmostEqual(t.p_lo, 0.52)
        self.assertAlmostEqual(t.p_hi, 0.68)

    def test_judge_error_correction_maps_true_edges_to_observed_edges(self):
        t = SequentialStopper(
            target=0.5,
            margin=0.1,
            total_samples=100,
            judge_false_negative_rate=0.10,
            judge_false_positive_rate=0.20,
        )
        self.assertAlmostEqual(t.p_lo, 0.4)
        self.assertAlmostEqual(t.p_hi, 0.6)
        self.assertAlmostEqual(t.q_lo, 0.48)
        self.assertAlmostEqual(t.q_hi, 0.62)
        self.assertEqual(t._k0, 48)
        self.assertEqual(t._k1, 62)

        for _ in range(10):
            res = t.update(0.62)
        payload = res.to_dict()
        self.assertAlmostEqual(payload['mean'], 0.6)
        self.assertAlmostEqual(payload['observed_mean'], 0.62)
        self.assertEqual(payload['p0'], 0.4)
        self.assertEqual(payload['p1'], 0.6)
        self.assertAlmostEqual(payload['q0'], 0.48)
        self.assertAlmostEqual(payload['q1'], 0.62)
        self.assertTrue(payload['judge_error_correction']['enabled'])
        self.assertEqual(payload['sprt_success_hypotheses'], [48, 62])

    def test_judge_error_correction_rejects_unusable_rates(self):
        with self.assertRaises(ValueError):
            SequentialStopper(
                target=0.5,
                margin=0.1,
                total_samples=100,
                judge_false_negative_rate=0.5,
                judge_false_positive_rate=0.5,
            )

    def test_risk_profiles(self):
        self.assertEqual(SequentialStopper.risk_levels('conservative'), (0.05, 0.10))
        self.assertEqual(SequentialStopper.risk_levels('balance'), (0.05, 0.20))
        self.assertEqual(SequentialStopper.risk_levels('balanced'), (0.05, 0.20))
        self.assertEqual(SequentialStopper.risk_levels('aggressive'), (0.10, 0.20))
        with self.assertRaises(ValueError):
            SequentialStopper.risk_levels('unknown')

    def test_estimated_margin_from_budget_and_risk(self):
        conservative_alpha, conservative_beta = SequentialStopper.risk_levels('conservative')
        aggressive_alpha, aggressive_beta = SequentialStopper.risk_levels('aggressive')
        small_budget = SequentialStopper.estimate_margin(
            target=0.6, total_samples=1000, sample_budget=100,
            alpha=conservative_alpha, beta=conservative_beta,
        )
        large_budget = SequentialStopper.estimate_margin(
            target=0.6, total_samples=1000, sample_budget=500,
            alpha=conservative_alpha, beta=conservative_beta,
        )
        aggressive = SequentialStopper.estimate_margin(
            target=0.6, total_samples=1000, sample_budget=100,
            alpha=aggressive_alpha, beta=aggressive_beta,
        )
        self.assertGreater(small_budget, large_budget)
        self.assertGreater(small_budget, aggressive)

    def test_rejects_out_of_range_observation(self):
        t = SequentialStopper(target=0.5, margin=0.05, total_samples=100)
        with self.assertRaises(ValueError):
            t.update(1.5)

    def test_fixed_population_sprt_likelihood(self):
        """Finite-N SPRT uses C(k,s)C(N-k,f) likelihood ratios."""
        t = SequentialStopper(target=0.5, margin=0.2, min_samples=1, total_samples=10)
        t.update(1.0)
        t.update(1.0)
        res = t.update(0.0)  # s=2, f=1; k0=3, k1=7
        expected = math.log((math.comb(7, 2) * math.comb(3, 1)) / (math.comb(3, 2) * math.comb(7, 1)))
        self.assertAlmostEqual(res.sprt_llr, expected)
        self.assertEqual(res.extra['sprt_success_hypotheses'], [3, 7])

    def test_fixed_population_sprt_impossible_under_null(self):
        """Once observed successes exceed k0, H0 likelihood is zero."""
        t = SequentialStopper(target=0.5, margin=0.3, strategy='sprt', min_samples=1, total_samples=10)
        t.update(1.0)
        t.update(1.0)
        res = t.update(1.0)  # k0=2, so three successes are impossible under H0.
        self.assertEqual(res.sprt_llr, math.inf)
        self.assertEqual(res.decision, 'above')

    def test_fixed_population_sprt_no_nan_when_both_edges_impossible(self):
        """If both edge-count hypotheses are impossible, SPRT stays neutral."""
        t = SequentialStopper(target=0.5, margin=0.05, strategy='sprt', min_samples=1, total_samples=10)
        # k0=4, k1=6. Ten observations with five successes and five failures
        # are impossible under both exact edge populations: H0 has too few
        # successes, H1 has too few failures.
        for x in (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0):
            res = t.update(x)
        self.assertFalse(math.isnan(res.sprt_llr))
        self.assertEqual(res.sprt_llr, 0.0)
        self.assertEqual(res.sprt_decision, 'undecided')

    def test_bayes_uses_finite_population_posterior(self):
        """Bayesian mode updates posterior mass over finite-population success counts."""
        t = SequentialStopper(target=0.5, margin=0.2, strategy='bayes', min_samples=1, total_samples=10)
        t.update(1.0)
        t.update(1.0)
        res = t.update(0.0)
        payload = res.to_dict()
        self.assertEqual(payload['interval_type'], 'bayesian_credible_interval')
        self.assertGreater(payload['bayes_posterior_above'], payload['bayes_posterior_below'])
        total_mass = (
            payload['bayes_posterior_below']
            + payload['bayes_posterior_within']
            + payload['bayes_posterior_above']
        )
        self.assertAlmostEqual(total_mass, 1.0)

    def test_bayes_can_stop_on_clear_signal(self):
        res = _run(0.95, strategy='bayes', n=1000, min_samples=20, max_samples=500, seed=1)
        self.assertEqual(res.decision, 'above')
        self.assertEqual(res.decided_by, 'bayes')
        self.assertGreaterEqual(res.extra['bayes_posterior_above'], 1.0 - res.extra['alpha'])

    def test_fixed_strategy_uses_wilson_at_budget(self):
        """Fixed mode runs to max_samples, then decides once from a Wilson interval."""
        t = SequentialStopper(target=0.5, margin=0.0, strategy='fixed', min_samples=1, max_samples=20,
                              total_samples=100)
        for _ in range(19):
            res = t.update(1.0)
            self.assertFalse(res.decided)
        res = t.update(1.0)
        self.assertEqual(res.decided_by, 'fixed')
        self.assertEqual(res.decision, 'above')
        self.assertGreater(res.ci_lower, 0.5)

    def test_fixed_strategy_borderline_when_wilson_contains_target(self):
        t = SequentialStopper(target=0.5, margin=0.0, strategy='fixed', min_samples=1, max_samples=20,
                              total_samples=100)
        for x in ([1.0, 0.0] * 10):
            res = t.update(x)
        self.assertEqual(res.decided_by, 'fixed')
        self.assertEqual(res.decision, 'within')
        self.assertLessEqual(res.ci_lower, 0.5)
        self.assertGreaterEqual(res.ci_upper, 0.5)

    def test_min_samples_floor(self):
        """No decision may be reported before min_samples, even when obvious."""
        t = SequentialStopper(target=0.5, margin=0.05, min_samples=50, total_samples=100)
        for _ in range(49):
            res = t.update(1.0)  # extreme, clearly 'above'
        self.assertFalse(res.decided)
        self.assertEqual(res.decision, 'undecided')

    def test_clearly_above(self):
        res = _run(0.95)
        self.assertEqual(res.decision, 'above')

    def test_clearly_below(self):
        res = _run(0.10)
        self.assertEqual(res.decision, 'below')

    def test_centre_resolves_within_not_directional(self):
        """A band-centred model must never be (mis)labelled above/below."""
        for seed in range(10):
            res = _run(0.60, strategy='agrappa', seed=seed)
            self.assertIn(res.decision, ('within',), f'seed={seed} -> {res.decision}')

    def test_no_false_directional_at_centre_across_seeds(self):
        """Anytime-valid: directional error at the band centre should be ~rare."""
        bad = 0
        for seed in range(40):
            res = _run(0.50, target=0.5, margin=0.05, strategy='agrappa', seed=seed)
            if res.decision in ('above', 'below'):
                bad += 1
        # With alpha=0.05 we expect this to be small; allow generous slack.
        self.assertLessEqual(bad, 4)

    def test_sprt_can_stop_before_agrappa(self):
        """SPRT may make a directional edge-likelihood call before the CS certifies the edge."""
        s = _run(0.80, target=0.6, margin=0.08, strategy='sprt', seed=1)
        a = _run(0.80, target=0.6, margin=0.08, strategy='agrappa', seed=1)
        self.assertEqual(s.decision, 'above')
        self.assertEqual(a.decision, 'above')
        self.assertLessEqual(s.n, a.n)

    def test_agrappa_estimation_stays_band_relative(self):
        """A band-centred model is 'within' under the strict estimator (no directional)."""
        res = _run(0.60, target=0.6, margin=0.08, strategy='agrappa', seed=3)
        self.assertEqual(res.decision, 'within')
        self.assertEqual(res.decided_by, 'agrappa')

    def test_verdict_is_always_definitive(self):
        """Verdict never reports 'undecided' — a finite run always yields a call."""
        for p in (0.2, 0.5, 0.78, 0.95):
            res = _run(p, target=0.8, margin=0.05, n=40, min_samples=30, seed=0)
            self.assertIn(res.verdict, ('PASS', 'FAIL', 'BORDERLINE'), f'p={p} -> {res.verdict}')

    def test_max_samples_budget_stops_borderline(self):
        """A near-target model hits the budget cap and stops BORDERLINE, not undecided."""
        # true 0.80 == target, wide zone [0.70, 0.90]: never resolves PASS/FAIL,
        # but the estimate sits solidly in the zone -> BORDERLINE at the cap.
        res = _run(
            0.80, target=0.8, margin=0.10, strategy='agrappa', n=10000,
            min_samples=30, max_samples=60, seed=0,
        )
        self.assertEqual(res.n, 60)                 # stopped at the budget
        self.assertEqual(res.decided_by, 'budget')
        self.assertEqual(res.verdict, 'BORDERLINE')  # estimate sits in the zone

    def test_undecided_verdict_uses_point_estimate(self):
        """No confident early call still gives a definitive PASS/FAIL from the score."""
        from evalscope.metrics.sequential_stopping import StoppingDecision

        def make(mean):
            return StoppingDecision(
                decision='undecided', decided_by=None, n=10, mean=mean,
                ci_lower=0.4, ci_upper=0.95, target_range=[0.75, 0.85], target=0.8,
                sprt_llr=0.0, sprt_decision='undecided', cs_decision='undecided',
            )
        self.assertEqual(make(0.62).verdict, 'FAIL')        # below the zone
        self.assertEqual(make(0.90).verdict, 'PASS')        # above the zone
        self.assertEqual(make(0.80).verdict, 'BORDERLINE')  # inside the ±margin zone
        self.assertEqual(make(0.75).verdict, 'BORDERLINE')  # exact lower edge is in the zone
        self.assertEqual(make(0.85).verdict, 'BORDERLINE')  # exact upper edge is in the zone

    def test_sprt_only_decides_direction(self):
        res = _run(0.95, strategy='sprt')
        self.assertEqual(res.decision, 'above')
        self.assertEqual(res.decided_by, 'sprt')

    def test_agrappa_only_can_say_within(self):
        res = _run(0.60, strategy='agrappa')
        self.assertEqual(res.decision, 'within')
        self.assertEqual(res.decided_by, 'agrappa')

    def test_confidence_sequence_contains_truth(self):
        """The anytime-valid CI should cover the true mean at the stop time."""
        covered = 0
        for seed in range(20):
            res = _run(0.80, seed=seed)
            if res.ci_lower <= 0.80 <= res.ci_upper:
                covered += 1
        self.assertGreaterEqual(covered, 18)  # ~ (1 - alpha) coverage

    def test_result_dict_serializable(self):
        import json
        res = _run(0.95)
        payload = res.to_dict()
        json.dumps(payload)  # must not raise
        self.assertEqual(payload['grey_zone']['lower'], payload['p0'])
        self.assertEqual(payload['grey_zone']['upper'], payload['p1'])
        self.assertEqual(payload['p_lo'], payload['target_range'][0])
        self.assertEqual(payload['p_hi'], payload['target_range'][1])
        self.assertIn('sprt_h0_success_rate', payload)
        self.assertIn('sprt_h1_success_rate', payload)


class TestSequentialEvaluatorLoop(unittest.TestCase):
    """Exercise the round loop / early-stop wiring with model + scoring mocked."""

    def _make_evaluator(self, early_stop_config):
        import types
        from evalscope.evaluator.sequential import SequentialEvaluator

        def get_sampling_segments(sample, subset):
            category = ev.benchmark.category_map.get(subset, 'default')
            return {'subset': subset, 'category': category}

        ev = object.__new__(SequentialEvaluator)  # bypass heavy __init__
        ev.benchmark = types.SimpleNamespace(
            use_batch_scoring=False,
            category_map={},
            get_sampling_segments=get_sampling_segments,
        )
        ev.benchmark_name = 'any_benchmark'
        ev.task_config = types.SimpleNamespace(eval_batch_size=4, ignore_errors=False)
        ev.early_stop_config = early_stop_config
        ev._warned_out_of_range = False
        ev._warned_no_score = False
        return ev

    def _context(self, n):
        from evalscope.evaluator.evaluator import _PoolContext, _WorkItem
        items = [_WorkItem(subset='release_latest', sample=MagicMockSample(i)) for i in range(n)]
        return _PoolContext(
            work_items=items,
            cached_scores_by_subset={},
            review_pending_by_subset={},
            model_prediction_dir='/tmp',
            total_cached=0,
        )

    def test_stops_early_on_clear_signal(self):
        from unittest.mock import MagicMock
        from evalscope.api.metric import SampleScore, Score

        ev = self._make_evaluator({
            'strategy': 'sprt', 'target_accuracy': 0.5,
            'risk_assessment': 'balance', 'min_samples': 20, 'seed': 1,
        })

        def fake_process(item, _dir):
            score = Score(value={'acc': 1.0}, main_score_name='acc')  # always pass -> clearly above
            return MagicMock(), SampleScore(score=score, sample_id=item.sample.idx)

        ev._process_work_item = fake_process
        ev._persist_result = lambda *a, **k: None

        results = ev._run_pool(self._context(500))

        self.assertTrue(ev._stop_info['stopped_early'])
        self.assertEqual(ev._stop_info['decision'], 'above')
        self.assertIn('grey_zone', ev._stop_info)
        self.assertEqual(ev._stop_info['grey_zone']['lower'], ev._stop_info['p0'])
        self.assertEqual(ev._stop_info['grey_zone']['upper'], ev._stop_info['p1'])
        self.assertLess(ev._stop_info['fresh_predictions'], 500)  # bailed out early
        ran = sum(len(v) for v in results.values())
        self.assertEqual(ran, ev._stop_info['fresh_predictions'])

    def test_fixed_mode_runs_exactly_to_prune_budget(self):
        from unittest.mock import MagicMock
        from evalscope.api.metric import SampleScore, Score

        ev = self._make_evaluator({
            'strategy': 'fixed', 'target_accuracy': 0.5, 'prune_ratio': 0.2,
            'risk_assessment': 'balance', 'min_samples': 1, 'seed': 1,
        })

        def fake_process(item, _dir):
            score = Score(value={'acc': 1.0}, main_score_name='acc')
            return MagicMock(), SampleScore(score=score, sample_id=item.sample.idx)

        ev._process_work_item = fake_process
        ev._persist_result = lambda *a, **k: None

        ev._run_pool(self._context(100))

        self.assertTrue(ev._stop_info['stopped_early'])
        self.assertEqual(ev._stop_info['sample_budget'], 20)
        self.assertEqual(ev._stop_info['fresh_predictions'], 20)
        self.assertEqual(ev._stop_info['samples_scored'], 20)
        self.assertEqual(ev._stop_info['decision'], 'above')
        self.assertEqual(ev._stop_info['decided_by'], 'fixed')

    def test_no_false_early_stop_when_nothing_pruned(self):
        """A verdict on the last sample (budget == work count) must not claim early stop."""
        from unittest.mock import MagicMock
        from evalscope.api.metric import SampleScore, Score

        ev = self._make_evaluator({  # prune_ratio 1.0 over 50 items -> budget fires on the last
            'strategy': 'sprt', 'target_accuracy': 0.5,
            'min_samples': 20, 'prune_ratio': 1.0, 'seed': 1,
        })

        counter = {'n': 0}

        def fake_process(item, _dir):  # alternate in processed order -> mean ~0.5, never resolves confidently
            v = float(counter['n'] % 2)
            counter['n'] += 1
            return MagicMock(), SampleScore(score=Score(value={'acc': v}, main_score_name='acc'),
                                            sample_id=item.sample.idx)

        ev._process_work_item = fake_process
        ev._persist_result = lambda *a, **k: None

        ev._run_pool(self._context(50))  # exactly max_samples items -> budget fires on the last
        self.assertEqual(ev._stop_info['samples_skipped'], 0)
        self.assertFalse(ev._stop_info['stopped_early'])     # nothing pruned
        self.assertEqual(ev._stop_info['verdict'], 'BORDERLINE')  # budget decision, mean in zone

    def test_cached_predictions_consumed_before_fresh_inference(self):
        """Review-only cached predictions resolve the run with zero fresh predictions."""
        from unittest.mock import MagicMock
        from evalscope.api.metric import SampleScore, Score
        from evalscope.evaluator.evaluator import _PoolContext, _WorkItem

        ev = self._make_evaluator({
            'strategy': 'sprt', 'target_accuracy': 0.5,
            'risk_assessment': 'balance', 'min_samples': 20, 'seed': 1,
        })

        fresh_calls = []

        def fake_process(item, _dir):
            if item.needs_predict:  # would be a real model call
                fresh_calls.append(item)
            return MagicMock(), SampleScore(score=Score(value={'acc': 1.0}, main_score_name='acc'))

        ev._process_work_item = fake_process
        ev._persist_result = lambda *a, **k: None

        # 200 review-only (cached predictions) that clearly resolve 'above', plus 300 fresh.
        review_only = [_WorkItem(subset='s', task_state=MagicMock()) for _ in range(200)]
        fresh = [_WorkItem(subset='s', sample=MagicMockSample(i)) for i in range(300)]
        ctx = _PoolContext(
            work_items=review_only + fresh, cached_scores_by_subset={},
            review_pending_by_subset={}, model_prediction_dir='/tmp', total_cached=0,
        )
        ev._run_pool(ctx)

        self.assertTrue(ev._stop_info['stopped_early'])
        self.assertEqual(ev._stop_info['fresh_predictions'], 0)  # decided from cache alone
        self.assertEqual(len(fresh_calls), 0)                    # no fresh model calls issued

    def test_batch_review_realigns_when_scores_compacted(self):
        """A dropped preliminary score must not shift scores onto the wrong task state."""
        import types
        from evalscope.api.metric import SampleScore, Score

        ev = self._make_evaluator({'target_accuracy': 0.5})
        task_states = [types.SimpleNamespace(sample_id=i) for i in range(3)]

        # review_subset drops the middle sample (id=1) and returns only ids 0 and 2,
        # in input order — the exact compaction that breaks a positional zip.
        def review_subset(subset, tss, review_fn):
            return [
                SampleScore(score=Score(value={'acc': 0.0}, main_score_name='acc'), sample_id=0),
                SampleScore(score=Score(value={'acc': 1.0}, main_score_name='acc'), sample_id=2),
            ]

        ev.batch_reviewer = types.SimpleNamespace(review_subset=review_subset)
        pairs = ev._batch_review('s', task_states)
        # Each score is paired with the task state of the SAME sample_id (not id=1).
        self.assertEqual([(ts.sample_id, sc.sample_id) for ts, sc in pairs], [(0, 0), (2, 2)])

    def test_batch_scoring_allowed(self):
        """Batch-scoring is supported (inference is still per-sample) -> no error."""
        ev = self._make_evaluator({'target_accuracy': 0.5})
        ev.benchmark.use_batch_scoring = True
        ev._validate_early_stop()  # must not raise

    def test_stops_early_with_batch_scoring(self):
        """Batch benchmarks review each round at its boundary and can stop early."""
        import types
        from evalscope.api.metric import SampleScore, Score

        ev = self._make_evaluator({
            'strategy': 'sprt', 'target_accuracy': 0.5,
            'risk_assessment': 'balance', 'min_samples': 20, 'seed': 1,  # window = eval_batch_size
        })
        ev.benchmark.use_batch_scoring = True
        # Prediction runs per-sample (task state carries its sample_id); review is
        # deferred (sample_score is None) until the round-end batch review.
        ev._process_work_item = lambda item, _dir: (types.SimpleNamespace(sample_id=item.sample.idx), None)
        ev._persist_result = lambda *a, **k: None
        # Batch reviewer returns a perfect score per task state (carrying its
        # sample_id, as the real reviewer does) -> clearly 'above'.
        ev.batch_reviewer = types.SimpleNamespace(
            review_subset=lambda subset, task_states, review_fn: [
                SampleScore(score=Score(value={'acc': 1.0}, main_score_name='acc'), sample_id=ts.sample_id)
                for ts in task_states
            ]
        )

        results = ev._run_pool(self._context(500))
        self.assertTrue(ev._stop_info['stopped_early'])
        self.assertEqual(ev._stop_info['decision'], 'above')
        self.assertLess(ev._stop_info['fresh_predictions'], 500)
        # batch-reviewed scores are collected for aggregation
        self.assertGreater(sum(len(v) for v in results.values()), 0)

    def test_missing_target_raises(self):
        """An early_stop block without target_accuracy is a hard error."""
        ev = self._make_evaluator({'prune_ratio': 0.5})  # no target_accuracy
        with self.assertRaises(ValueError):
            ev._validate_early_stop()

    def test_valid_config_passes_validation(self):
        ev = self._make_evaluator({'target_accuracy': 0.5})
        self.assertEqual(ev._read_config()['strategy'], 'sprt')
        ev._validate_early_stop()  # must not raise

    def test_method_alias_for_strategy(self):
        ev = self._make_evaluator({'target_accuracy': 0.5, 'method': 'fixed'})
        self.assertEqual(ev._read_config()['strategy'], 'fixed')
        ev._validate_early_stop()

    def test_bayes_config_passes_prior_to_stopper(self):
        ev = self._make_evaluator({
            'target_accuracy': 0.5,
            'strategy': 'bayes',
            'bayes_prior_alpha': 2.0,
            'bayes_prior_beta': 3.0,
        })
        cfg = ev._read_config()
        self.assertEqual(cfg['strategy'], 'bayes')
        self.assertEqual(cfg['bayes_prior_alpha'], 2.0)
        self.assertEqual(cfg['bayes_prior_beta'], 3.0)
        ev._validate_early_stop()

    def test_judge_error_config_aliases(self):
        ev = self._make_evaluator({'target_accuracy': 0.5, 'fnr': 0.1, 'fpr': 0.2})
        cfg = ev._read_config()
        self.assertEqual(cfg['judge_false_negative_rate'], 0.1)
        self.assertEqual(cfg['judge_false_positive_rate'], 0.2)
        ev._validate_early_stop()

    def test_judge_error_config_rejects_no_signal(self):
        ev = self._make_evaluator({
            'target_accuracy': 0.5,
            'judge_false_negative_rate': 0.6,
            'judge_false_positive_rate': 0.4,
        })
        with self.assertRaises(ValueError):
            ev._validate_early_stop()

    def test_bayes_strategy_aliases(self):
        ev = self._make_evaluator({'target_accuracy': 0.5, 'strategy': 'baysian'})
        self.assertEqual(ev._read_config()['strategy'], 'bayes')
        ev._validate_early_stop()

    def test_out_of_range_score_clamped_and_warned(self):
        from evalscope.api.metric import SampleScore, Score
        ev = self._make_evaluator({'target_accuracy': 0.5})
        ss = SampleScore(score=Score(value={'acc': 1.5}, main_score_name='acc'))
        self.assertEqual(ev._extract_accuracy(ss), 1.0)
        self.assertTrue(ev._warned_out_of_range)
        # in-range value never trips the warning
        ev2 = self._make_evaluator({'target_accuracy': 0.5})
        ev2._extract_accuracy(SampleScore(score=Score(value={'acc': 0.5}, main_score_name='acc')))
        self.assertFalse(ev2._warned_out_of_range)

    def test_stratified_sampling_interleaves_existing_categories(self):
        from evalscope.evaluator.evaluator import _WorkItem

        ev = self._make_evaluator({
            'target_accuracy': 0.5,
            'sampling': {
                'strategy': 'stratified',
                'dimensions': ['category'],
                'seed': 1,
            },
        })
        ev.benchmark.category_map = {'a_1': 'A', 'a_2': 'A', 'b': 'B'}
        items = (
            [_WorkItem(subset='a_1', sample=MagicMockSample(i)) for i in range(6)]
            + [_WorkItem(subset='a_2', sample=MagicMockSample(i)) for i in range(2)]
            + [_WorkItem(subset='b', sample=MagicMockSample(i)) for i in range(4)]
        )

        ordered = ev._order_work_items(items, ev._read_config(), random.Random(1))
        prefix = Counter(ev._item_stratum(item, ev._read_config()) for item in ordered[:6])

        self.assertEqual(prefix, Counter({'category=A': 4, 'category=B': 2}))

    def test_stratified_sampling_supports_legacy_aliases(self):
        ev = self._make_evaluator({
            'target_accuracy': 0.5,
            'sampling_strategy': 'stratified',
            'stratify_by': 'category',
            'seed': 7,
        })
        cfg = ev._read_config()

        self.assertEqual(cfg['sampling']['strategy'], 'stratified')
        self.assertEqual(cfg['sampling']['dimensions'], ['category'])
        self.assertEqual(cfg['sampling']['seed'], 7)

    def test_stratified_sampling_config_validation(self):
        ev = self._make_evaluator({'target_accuracy': 0.5, 'sampling_strategy': 'unknown'})
        with self.assertRaises(ValueError):
            ev._validate_early_stop()

        ev = self._make_evaluator({'target_accuracy': 0.5, 'sampling': {'strategy': 'stratified', 'dimensions': []}})
        with self.assertRaises(ValueError):
            ev._validate_early_stop()

        ev = self._make_evaluator({
            'target_accuracy': 0.5,
            'sampling': {
                'strategy': 'stratified',
                'dimensions': ['category'],
                'allocation': 'equal',
            },
        })
        with self.assertRaises(ValueError):
            ev._validate_early_stop()


class TestGenericRouting(unittest.TestCase):
    """`early_stop` in dataset_args routes ANY benchmark to the sequential evaluator."""

    def _make(self, dataset, dataset_args, tmpdir):
        import evalscope  # ensure registries are populated
        from evalscope.config import TaskConfig
        from evalscope.api.registry import create_evaluator, get_benchmark
        from evalscope.utils.io_utils import OutputsStructure

        cfg = TaskConfig(model='dummy', datasets=[dataset], dataset_args=dataset_args)
        benchmark = get_benchmark(dataset, cfg)
        outputs = OutputsStructure(outputs_dir=tmpdir)
        return create_evaluator(benchmark=benchmark, model=None, outputs=outputs, task_config=cfg)

    def test_arbitrary_benchmark_with_early_stop(self):
        import tempfile
        from evalscope.evaluator.sequential import SequentialEvaluator
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._make(
                'gsm8k',
                {'gsm8k': {'early_stop': {'target_accuracy': 0.8, 'prune_ratio': 0.3}}},
                tmp,
            )
        self.assertIsInstance(ev, SequentialEvaluator)
        self.assertEqual(ev.early_stop_config['target_accuracy'], 0.8)

    def test_empty_early_stop_block_fails_fast(self):
        """An empty `early_stop` block must hard-error, not silently run full eval."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._make('gsm8k', {'gsm8k': {'early_stop': {}}}, tmp)

    def test_arbitrary_benchmark_without_early_stop_uses_default(self):
        import tempfile
        from evalscope.evaluator.evaluator import DefaultEvaluator
        from evalscope.evaluator.sequential import SequentialEvaluator
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._make('gsm8k', {}, tmp)
        self.assertIsInstance(ev, DefaultEvaluator)
        self.assertNotIsInstance(ev, SequentialEvaluator)


class TestSequentialReportFormat(unittest.TestCase):

    def test_report_metadata_roundtrip_and_jsonl_sidecar_ignored(self):
        import tempfile
        from evalscope.report import Category, Metric, Report, Subset, get_report_list

        report = Report(
            name='eval',
            dataset_name='toy',
            model_name='model',
            metrics=[Metric(name='acc', categories=[Category(subsets=[Subset(name='default', score=1.0, num=3)])])],
            metadata={'sequential_stop': {'verdict': 'PASS', 'sample_budget': 2}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            report.to_json(f'{tmp}/toy.json')
            with open(f'{tmp}/toy_sequential_stop.jsonl', 'w', encoding='utf-8') as f:
                f.write('{"verdict": "PASS"}\n')

            reports = get_report_list([tmp])

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].metadata['sequential_stop']['sample_budget'], 2)


class TestShouldStopHook(unittest.TestCase):
    """The executor's should_stop hook halts submission without cancelling in-flight work."""

    def test_should_stop_halts_submission_no_overshoot_single_worker(self):
        from evalscope.utils.function_utils import run_in_threads_with_progress
        processed = []
        # max_workers=1 => no in-flight overshoot; should stop at exactly 5.
        run_in_threads_with_progress(
            list(range(100)), lambda x: x, desc='t', max_workers=1,
            on_result=lambda x, r: processed.append(r),
            should_stop=lambda: len(processed) >= 5,
        )
        self.assertEqual(len(processed), 5)

    def test_overshoot_bounded_by_workers(self):
        from evalscope.utils.function_utils import run_in_threads_with_progress
        processed = []
        workers = 4
        run_in_threads_with_progress(
            list(range(200)), lambda x: x, desc='t', max_workers=workers,
            on_result=lambda x, r: processed.append(r),
            should_stop=lambda: len(processed) >= 10,
        )
        # Stops near 10; never the full 200, overshoot at most ~one window.
        self.assertGreaterEqual(len(processed), 10)
        self.assertLess(len(processed), 10 + workers + 1)

    def test_no_should_stop_runs_all(self):
        from evalscope.utils.function_utils import run_in_threads_with_progress
        out = run_in_threads_with_progress(list(range(20)), lambda x: x, desc='t', max_workers=4)
        self.assertEqual(len(out), 20)


class MagicMockSample:
    def __init__(self, idx):
        self.idx = idx


if __name__ == '__main__':
    unittest.main()
