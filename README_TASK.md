# Sequential Early-Stopping Evaluation (SPRT + aGRAPA)

Stop an evaluation once the model's accuracy is statistically resolved against
a target band, instead of running every sample. This is a generic, opt-in eval
behavior: add an `early_stop` block to any dataset whose per-sample score exposes
an accuracy-like scalar in `[0, 1]` (`acc` or the benchmark main score).

## TL;DR

```bash
DATASET_ARGS='{
  "gsm8k": {
    "early_stop": {"target_accuracy": 0.8, "prune_ratio": 0.3, "risk_assessment": "balance"}
  }
}'

evalscope eval --model <model> --datasets gsm8k \
  --dataset-args "$DATASET_ARGS" \
  --output ./results_pruned/
```

`target_accuracy` is the go/no-go bar. `prune_ratio` is the planned evaluation
budget fraction: `0.3` means derive the tolerance band for an estimate after
about `ceil(0.3 * dataset_size)` scored samples. Confident models can still stop
earlier; unresolved borderline models stop at that budget and report a
point-estimate verdict.

## User Config

Place under `dataset_args["<dataset>"]["early_stop"]`.

| Key | Default | Meaning |
|-----|---------|---------|
| `target_accuracy` | required | Target accuracy bar in `(0, 1)`. Missing is a hard error. |
| `prune_ratio` | `1.0` | Fraction of the full dataset to score before budget fallback. Must be in `(0, 1]`. |
| `risk_assessment` | `balance` | Risk profile used to derive the margin from target and budget. |
| `strategy` | `sprt` | `sprt` \| `bayes` \| `agrappa` \| `fixed`. `method` is accepted as an alias. |
| `min_samples` | `30` | No stop may fire before this many scored samples. Clamped to the sample budget. |
| `seed` | `42` | Shuffle seed controlling sample evaluation order. |
| `sampling` | `{"strategy": "uniform"}` | Sample-ordering block. Use `stratified` for proportional segment sampling. |
| `bayes_prior_alpha` | `1.0` | Beta prior alpha for `strategy="bayes"`. |
| `bayes_prior_beta` | `1.0` | Beta prior beta for `strategy="bayes"`. |
| `judge_false_negative_rate` | `0.0` | LLM-judge false-negative rate. Aliases: `judge_fnr`, `fnr`. |
| `judge_false_positive_rate` | `0.0` | LLM-judge false-positive rate. Aliases: `judge_fpr`, `fpr`. |

Risk profiles:

`bayesian` and `baysian` are accepted as aliases for `bayes`; reports normalize
the strategy name to `bayes`.

| Risk Assessment | Type I | Type II |
|-----------------|--------|---------|
| `conservative` | 5% | 10% |
| `balance` | 5% | 20% |
| `aggressive` | 10% | 20% |

`balanced` is accepted as an alias for `balance`.

The evaluator derives the internal band half-width with a finite-population
normal approximation:

```text
sample_budget = ceil(prune_ratio * total_size)
margin ~= (z_(1-type1) + z_(1-type2)) *
          sqrt(target_accuracy * (1 - target_accuracy) *
               (total_size - sample_budget) / (total_size - 1) / sample_budget)
```

The band is then `[target_accuracy - margin, target_accuracy + margin]`, clipped
inside `(0, 1)`. Smaller budgets and more conservative risk profiles produce a
wider borderline band; larger budgets produce a tighter band.

### Judge-Bias Correction

For LLM-judge benchmarks, the observed score is the judge verdict, not the
latent true correctness. Optional false-negative and false-positive rates apply
the observation model:

```text
q = fpr + (1 - fnr - fpr) * p
```

where `p` is true accuracy and `q` is the observed judge-positive rate. The
reported `p0`/`p1` remain the true lower/upper grey-zone edges. The engine also
reports `q0`/`q1`, the observed edges actually used by SPRT, Bayes, aGRAPA, and
fixed Wilson decisions. Mean and CI fields are mapped back to corrected true
accuracy; `observed_mean` and `observed_ci_*` preserve the raw judge-verdict
scale.

### Segment-Aware Sampling

By default, sequential evaluation keeps the old behavior: it uniformly shuffles
all work items and then applies the stopping rule to that stream. For
heterogeneous benchmarks, use proportional stratified sampling to keep the
early stream representative of existing dataset segments without changing
scoring or aggregation logic.

Recommended format:

```json
{
  "early_stop": {
    "target_accuracy": 0.8,
    "prune_ratio": 0.3,
    "sampling": {
      "strategy": "stratified",
      "dimensions": ["category"],
      "allocation": "proportional",
      "min_per_stratum": 1,
      "seed": 42
    }
  }
}
```

`allocation` currently supports `proportional`, meaning the evaluator builds a
stream whose strata appear in roughly the same proportions as the loaded
dataset. `min_per_stratum` optionally front-loads a small coverage floor before
the proportional schedule continues. The stop calculation still sees only the
score sequence; this only changes which samples are submitted first.

Built-in dimensions:

| Dimension | Source |
|-----------|--------|
| `subset` | EvalScope dataset subset name, for example `release_latest` in LiveCodeBench. |
| `category` | Adapter `category_map` value for the subset, for example MMMU's domain category. |
| `metadata.<field>` | A value stored on `Sample.metadata` or cached task-state metadata. |

Adapters can expose benchmark-specific dimensions by overriding
`DataAdapter.get_sampling_segments`. This keeps the sampling config generic
while letting each benchmark define its own segmentation contract:

```python
from typing import Any, Dict, Optional

from evalscope.api.benchmark import DataAdapter
from evalscope.api.dataset import Sample


class CustomAdapter(DataAdapter):

    def get_sampling_segments(self, sample: Optional[Sample], subset: str) -> Dict[str, Any]:
        segments = super().get_sampling_segments(sample=sample, subset=subset)
        metadata = sample.metadata if sample else {}
        segments["difficulty"] = metadata.get("difficulty", "unknown")
        segments["question_type"] = metadata.get("question_type", "unknown")
        return segments
```

Then sample over one or more dimensions:

```json
{
  "sampling": {
    "strategy": "stratified",
    "dimensions": ["category", "difficulty"]
  }
}
```

Multiple dimensions form a joint stratum such as
`category=Science|difficulty=hard`. Legacy aliases are still accepted for short
configs: `sampling_strategy="stratified"` and `stratify_by="category"`.

## Decision Logic

Two engines are maintained after each scored sample:

- **aGRAPA confidence sequence**: anytime-valid confidence interval. It certifies
  `above` only when `ci_lower > upper_band`, `below` only when
  `ci_upper < lower_band`, and `within` when the whole CI is inside the band.
- **Fixed-N SPRT likelihood screen**: uses the known full dataset size `N` and
  fixed observed-success counts `k0 = floor(N * q0)`,
  `k1 = ceil(N * q1)`, where `q0/q1` equal `p0/p1` when judge correction is not
  configured. It asks whether the observed data looks much closer to one edge
  hypothesis than the other. This is a fast screen, not a confidence-interval
  certificate that accuracy is outside the band.
- **Bayesian finite-population posterior**: starts from a `Beta(alpha, beta)`
  prior over the full dataset success rate, induces a prior over total success
  count `K`, and updates with the same sampling-without-replacement likelihood
  used by SPRT. It reports posterior mass below, inside, and above the grey zone.

Strategies:

| Value | Use For | Behavior |
|-------|---------|----------|
| `agrappa` | Strict accuracy location | CS only; directional calls exclude the band edge. |
| `sprt` | Directional screen | Fast edge-likelihood screen only; never returns `within`; capped by `prune_ratio`. |
| `bayes` | Bayesian go/no-go screen | Posterior mass over finite-population success count. |
| `fixed` | Simple fixed-budget estimate | Runs to `ceil(prune_ratio * N)`, then decides once with a Wilson CI. |

For `sprt`, `above`/`below` means the observed sample is more consistent with
the upper/lower edge hypothesis than the opposite edge by the configured
evidence cutoff. It does not mean the reported CI excludes the band.

The plain-language verdict is always three-way:

- `PASS`: above the band.
- `FAIL`: below the band.
- `BORDERLINE`: inside the band, including exact band edges.

## Examples

CLI:

```bash
DATASET_ARGS='{
  "live_code_bench": {
    "subset_list": ["release_latest"],
    "early_stop": {
      "strategy": "sprt",
      "target_accuracy": 0.60,
      "prune_ratio": 0.3,
      "risk_assessment": "balance",
      "min_samples": 30
    }
  }
}'

evalscope eval --model qwen-plus \
  --api-url $OPENAI_API_BASE_URL --api-key $OPENAI_API_KEY --eval-type openai_api \
  --datasets live_code_bench \
  --dataset-args "$DATASET_ARGS" \
  --output ./results_pruned/
```

Python:

```python
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

early_stop = {
    "strategy": "sprt",
    "target_accuracy": 0.80,
    "prune_ratio": 0.30,
    "risk_assessment": "balance",
    "min_samples": 30,
}

task_cfg = TaskConfig(
    model="qwen-plus",
    api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="...",
    eval_type=EvalType.OPENAI_API,
    datasets=["gsm8k", "live_code_bench"],
    dataset_args={
        "gsm8k": {"early_stop": early_stop},
        "live_code_bench": {
            "subset_list": ["release_latest"],
            "early_stop": {**early_stop, "target_accuracy": 0.60},
        },
    },
)
run_task(task_cfg=task_cfg)
```

A runnable copy lives at `examples/benchmarks/early_stop_sequential.py`.

## Output

The standard EvalScope report remains a normal `Report`. The score is computed
over the samples that actually ran, and the stop details are embedded under
`metadata.sequential_stop`. A JSONL copy is also written to
`reports/<model>/<dataset>_sequential_stop.jsonl`; the `.jsonl` extension keeps
EvalScope's report scanner from treating it as another benchmark report.

```json
{
    "stopped_early": true,
    "prune_ratio": 0.3,
    "sample_budget": 396,
    "risk_assessment": "balance",
    "fresh_predictions": 64,
    "free_samples": 0,
    "samples_skipped": 1255,
    "samples_scored": 64,
    "samples_total": 1319,
    "verdict": "PASS",
    "decision_explanation": "Decision is certified by the anytime-valid confidence sequence.",
    "decision": "above",
    "decided_by": "agrappa",
    "n": 64,
    "mean": 0.8906,
    "ci_lower": 0.8100,
    "ci_upper": 0.9500,
    "target": 0.8,
    "target_range": [0.758, 0.842],
    "grey_zone": {
        "lower": 0.758,
        "upper": 0.842,
        "center": 0.8,
        "lower_margin": 0.042,
        "upper_margin": 0.042
    },
    "p_lo": 0.758,
    "p_hi": 0.842,
    "p0": 0.758,
    "p1": 0.842,
    "sprt_llr": 12.9,
    "sprt_decision": "above",
    "cs_decision": "above",
    "strategy": "agrappa",
    "alpha": 0.05,
    "beta": 0.20,
    "margin": 0.042,
    "min_samples": 30,
    "sampling": {
        "strategy": "stratified",
        "dimensions": ["category"],
        "allocation": "proportional",
        "min_per_stratum": 1,
        "seed": 42,
        "strata": {
            "category=Business": 300,
            "category=Science": 180
        },
        "budget_strata": {
            "category=Business": 90,
            "category=Science": 54
        }
    },
    "total_samples": 1319,
    "sprt_success_hypotheses": [1000, 1111],
    "sprt_h0_success_rate": 0.7582,
    "sprt_h1_success_rate": 0.8423,
    "interval_type": "confidence_sequence",
    "sprt_evidence_cutoffs": [-1.558, 2.773],
    "log_wealth_threshold": 2.996
}
```

Key fields:

| Field | Meaning |
|-------|---------|
| `stopped_early` | `true` only if work was actually skipped. |
| `samples_skipped` | Work items left unrun because the test stopped. |
| `fresh_predictions` | Model inferences actually run. |
| `free_samples` | Scores consumed from cache before inference. |
| `sample_budget` | Budget cap `ceil(prune_ratio * samples_total)`. |
| `sampling` | Normalized sample-ordering config plus loaded and budgeted stratum counts. |
| `risk_assessment`, `alpha`, `beta`, `margin` | The risk profile and derived internal band. |
| `grey_zone`, `target_range` | The derived borderline / indifference zone `[lower_band, upper_band]`. |
| `p0`, `p1` | SPRT edge probabilities for the lower and upper grey-zone hypotheses. |
| `q0`, `q1` | Observed judge-positive edge probabilities after optional judge-bias correction. |
| `sprt_success_hypotheses` | Rounded finite-population success counts `[k0, k1]` used by SPRT. |
| `sprt_h0_success_rate`, `sprt_h1_success_rate` | Actual rounded success rates `k0 / N` and `k1 / N`. |
| `observed_mean`, `observed_ci_lower`, `observed_ci_upper` | Raw judge-verdict estimate/interval before correction. |
| `judge_error_correction` | Configured `fnr`, `fpr`, transform scale, and formula. |
| `bayes_prior`, `bayes_posterior_*` | Prior and posterior mass below / within / above the grey zone. |
| `interval_type` | Confidence sequence, Wilson interval, or Bayesian credible interval. |
| `decision`, `decided_by` | Machine decision and engine that fired. |
| `decision_explanation` | Plain-English interpretation, especially important for `sprt` decisions. |
| `verdict` | Ship-oriented `PASS` / `FAIL` / `BORDERLINE`. |

## Notes

- Scores are assumed accuracy-like and are clamped into `[0, 1]` with a warning.
- Multiple subsets are pooled into one stream. For heterogeneous benchmarks,
  restrict to one subset when the target should apply to a specific slice.
- Batch-scoring benchmarks are supported, but the stop check happens after each
  `eval_batch_size` review window rather than after each sample.
- Cached scores are fed before new inference, so a cached run can resolve with
  zero fresh predictions.

## Implementation Map

| File | Role |
|------|------|
| `evalscope/metrics/sequential_stopping.py` | `SequentialStopper` decision engine. |
| `evalscope/evaluator/sequential.py` | Sequential evaluator loop, budget handling, sidecar. |
| `evalscope/api/registry.py` | Routes datasets with an `early_stop` block to the sequential evaluator. |
| `tests/test_sequential_stopping.py` | Unit and integration tests. |
| `examples/benchmarks/early_stop_sequential.py` | Runnable example. |

## Forked from
SHA of evalscope commit i fork from e1b4d09aa4a5bbdab7fa5eeaf567d74c8469a6e7