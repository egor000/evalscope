# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Generic sequential early-stopping evaluator.

Drop-in replacement for :class:`DefaultEvaluator` that processes samples in
mini-batch *rounds* and consults a :class:`SequentialStopper`
after every round. As soon as the model's accuracy is resolved against the
target band derived from ``target_accuracy``, ``prune_ratio``, and
``risk_assessment``, the remaining samples are left unrun — saving inference
cost while retaining an anytime-valid guarantee.

This evaluator is **benchmark-agnostic**: it works with any benchmark whose
per-sample score exposes a scalar in ``[0, 1]`` (the ``acc`` metric, or the
benchmark's main score clamped to ``[0, 1]``). It is selected automatically by
:func:`evalscope.api.registry.create_evaluator` whenever an ``early_stop`` block
is present in a dataset's ``--dataset-args``::

    --datasets gsm8k \\
    --dataset-args '{"gsm8k": {"early_stop": {"target_accuracy": 0.8, "prune_ratio": 0.3}}}'

Supported ``early_stop`` keys:

* ``strategy``        – ``'sprt'`` | ``'bayes'`` | ``'agrappa'`` | ``'fixed'`` (default ``'sprt'``)
* ``target_accuracy`` – target accuracy bar in ``(0, 1)``; required to enable stopping
* ``prune_ratio``     – fraction of the dataset to evaluate before budget fallback
* ``risk_assessment`` – ``'conservative'`` | ``'balance'`` | ``'aggressive'``
* ``min_samples``     – minimum samples before a stop may fire (default ``30``)
* ``seed``            – shuffle seed controlling sample evaluation order (default ``42``)
* ``sampling``        – optional sample-ordering config:
                        ``strategy='uniform'|'stratified'``, ``dimensions=[...]``
* ``bayes_prior_alpha`` / ``bayes_prior_beta`` – Beta prior for ``strategy='bayes'`` (default ``1``)
* ``judge_false_negative_rate`` / ``judge_false_positive_rate`` – optional judge-bias correction rates

Per-sample-scored benchmarks (including per-sample LLM-judge ones such as
``aa_lcr``) check after **every** result and stop immediately, so overshoot is
bounded only by ``eval_batch_size`` concurrency. The only exception is the
handful of benchmarks that set ``use_batch_scoring`` (their metric is computed
in batches with no meaningful per-sample value): those are reviewed in
``eval_batch_size`` windows, checking the rule after each window.
"""

import json
import math
import os
import random
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.registry import register_evaluator
from evalscope.constants import HEARTBEAT_INTERVAL_SEC
from evalscope.evaluator.evaluator import DefaultEvaluator, _PoolContext, _WorkItem
from evalscope.metrics.sequential_stopping import SequentialStopper
from evalscope.utils.function_utils import run_in_threads_with_progress
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_evaluator('sequential')
class SequentialEvaluator(DefaultEvaluator):
    """Evaluator that stops early once accuracy is resolved against a target band."""

    def __init__(self, benchmark, model, outputs, task_config):
        super().__init__(benchmark=benchmark, model=model, outputs=outputs, task_config=task_config)
        # Read the ``early_stop`` block now, before run.py overwrites dataset_args.
        ds_args = (task_config.dataset_args.get(benchmark.name) or {}) if task_config else {}
        self.early_stop_config: dict = ds_args.get('early_stop') or {}
        self._sequential_token_usage = {
            'fresh_predictions': {
                'requests': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
            },
        }
        self._warned_out_of_range = False
        self._warned_no_score = False
        # Fail fast on unsupported / misconfigured setups so no inference is wasted.
        self._validate_early_stop()

    def _read_config(self) -> dict:
        """Resolve sequential-test settings from :attr:`early_stop_config`."""
        cfg = self.early_stop_config or {}
        strategy = SequentialStopper.normalize_strategy(cfg.get('strategy', cfg.get('method', 'sprt')))
        sampling = self._read_sampling_config(cfg)
        return {
            'strategy': strategy,
            'target_accuracy': cfg.get('target_accuracy', None),
            'risk_assessment': cfg.get('risk_assessment', 'balance'),
            'min_samples': int(cfg.get('min_samples', 30)),
            'prune_ratio': float(cfg.get('prune_ratio', 1.0)),
            'seed': sampling['seed'],
            'sampling': sampling,
            'sampling_strategy': sampling['strategy'],
            'stratify_by': sampling['dimensions'][0] if sampling['dimensions'] else '',
            'bayes_prior_alpha': float(cfg.get('bayes_prior_alpha', 1.0)),
            'bayes_prior_beta': float(cfg.get('bayes_prior_beta', 1.0)),
            'judge_false_negative_rate': float(
                cfg.get('judge_false_negative_rate', cfg.get('judge_fnr', cfg.get('fnr', 0.0)))
            ),
            'judge_false_positive_rate': float(
                cfg.get('judge_false_positive_rate', cfg.get('judge_fpr', cfg.get('fpr', 0.0)))
            ),
        }

    def _read_sampling_config(self, cfg: dict) -> dict:
        """Normalize short sampling aliases and the richer sampling block."""
        raw = cfg.get('sampling') or {}
        if raw and not isinstance(raw, dict):
            raise ValueError(f"sampling must be a dict when provided, got {type(raw).__name__}")

        strategy = raw.get('strategy', cfg.get('sampling_strategy', 'uniform'))
        strategy = str(strategy).lower().replace('-', '_')
        if strategy == 'proportional_stratified':
            strategy = 'stratified'

        dimensions = raw.get('dimensions', raw.get('dimension', cfg.get('stratify_by', ['subset'])))
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        dimensions = [str(dim).strip() for dim in dimensions]

        allocation = str(raw.get('allocation', 'proportional')).lower().replace('-', '_')
        return {
            'strategy': strategy,
            'dimensions': dimensions,
            'allocation': allocation,
            'min_per_stratum': int(raw.get('min_per_stratum', 0)),
            'seed': int(raw.get('seed', cfg.get('seed', 42))),
        }

    def _validate_early_stop(self) -> None:
        """Eagerly reject configurations that cannot save inference cost.

        Raised here (at construction, before the model is loaded) rather than
        falling back to a full evaluation — silently running every sample would
        defeat the purpose of early stopping.
        """
        cfg = self._read_config()
        if cfg['target_accuracy'] is None:
            raise ValueError(
                f"Early stopping for '{self.benchmark_name}' requires 'target_accuracy' "
                "(the band centre, in (0, 1)) in the 'early_stop' config."
            )
        if not (0.0 < cfg['prune_ratio'] <= 1.0):
            raise ValueError(f"prune_ratio must be in (0, 1], got {cfg['prune_ratio']}")
        sampling = cfg['sampling']
        if sampling['strategy'] not in ('uniform', 'stratified'):
            raise ValueError(
                "sampling.strategy must be one of: 'uniform', 'stratified'; "
                f"got {sampling['strategy']!r}"
            )
        if not sampling['dimensions'] or any(not dim for dim in sampling['dimensions']):
            raise ValueError('sampling.dimensions must contain at least one non-empty dimension.')
        if sampling['allocation'] != 'proportional':
            raise ValueError(f"sampling.allocation must be 'proportional', got {sampling['allocation']!r}")
        if sampling['min_per_stratum'] < 0:
            raise ValueError(f"sampling.min_per_stratum must be >= 0, got {sampling['min_per_stratum']}")
        if cfg['bayes_prior_alpha'] <= 0.0 or cfg['bayes_prior_beta'] <= 0.0:
            raise ValueError(
                f"bayes_prior_alpha and bayes_prior_beta must be > 0, got "
                f"{cfg['bayes_prior_alpha']}, {cfg['bayes_prior_beta']}"
            )
        if not (0.0 <= cfg['judge_false_negative_rate'] < 1.0):
            raise ValueError(
                f"judge_false_negative_rate must be in [0, 1), got {cfg['judge_false_negative_rate']}"
            )
        if not (0.0 <= cfg['judge_false_positive_rate'] < 1.0):
            raise ValueError(
                f"judge_false_positive_rate must be in [0, 1), got {cfg['judge_false_positive_rate']}"
            )
        if cfg['judge_false_negative_rate'] + cfg['judge_false_positive_rate'] >= 1.0:
            raise ValueError(
                'judge_false_negative_rate + judge_false_positive_rate must be < 1, got '
                f"{cfg['judge_false_negative_rate']} + {cfg['judge_false_positive_rate']}"
            )
        alpha, beta = SequentialStopper.risk_levels(cfg['risk_assessment'])
        # Dataset size is only known after loading, so margin is derived in
        # _run_pool. Fixed mode does not use an indifference margin.
        margin = 0.0 if cfg['strategy'] == 'fixed' else 1e-6
        SequentialStopper.validate_config(
            target=cfg['target_accuracy'], margin=margin, alpha=alpha, beta=beta, strategy=cfg['strategy']
        )

    def _run_pool(self, context: _PoolContext) -> Dict[str, List[Tuple[TaskState, Optional[SampleScore]]]]:
        """Evaluate with early stopping; model inference (the saved cost) is per-sample.

        Two execution modes, chosen by how the benchmark is scored:

        * **Per-sample scoring → streaming** (:meth:`_run_streaming`): one rolling
          window over all samples; the moment the test decides we stop submitting
          new work, so overshoot is bounded only by the in-flight window
          (``eval_batch_size``). This covers per-sample LLM-judge benchmarks too.
        * **Batch scoring → windowed** (:meth:`_run_windowed`): the few benchmarks
          whose metric is computed in batches (``use_batch_scoring``) have no
          per-sample score, so predictions run in ``eval_batch_size`` windows, each
          window is batch reviewed, then the rule is checked.

        Already-available scores (cache, batch review-pending) are fed first, at
        zero inference cost. Collected scores are reused at aggregation
        (:meth:`_aggregate_scores`) so nothing is reviewed twice. Config is
        validated in :meth:`__init__`.
        """
        cfg = self._read_config()
        is_batch = self.benchmark.use_batch_scoring

        samples_total = len(context.work_items) + context.total_cached
        alpha, beta = SequentialStopper.risk_levels(cfg['risk_assessment'])
        sample_budget = max(1, math.ceil(cfg['prune_ratio'] * samples_total))
        effective_min_samples = min(cfg['min_samples'], sample_budget)
        if cfg['strategy'] == 'fixed':
            margin = 0.0
        else:
            margin = SequentialStopper.estimate_margin(
                target=cfg['target_accuracy'],
                total_samples=samples_total,
                sample_budget=sample_budget,
                alpha=alpha,
                beta=beta,
            )

        stopper = SequentialStopper(
            target=cfg['target_accuracy'], margin=margin, alpha=alpha, beta=beta,
            strategy=cfg['strategy'], min_samples=effective_min_samples, max_samples=sample_budget,
            total_samples=samples_total, bayes_prior_alpha=cfg['bayes_prior_alpha'],
            bayes_prior_beta=cfg['bayes_prior_beta'],
            judge_false_negative_rate=cfg['judge_false_negative_rate'],
            judge_false_positive_rate=cfg['judge_false_positive_rate'],
        )
        mode = f'windowed(window={self.task_config.eval_batch_size})' if is_batch else 'streaming'
        logger.info(
            f"Sequential early stopping enabled (strategy={cfg['strategy']}, "
            f"target={cfg['target_accuracy']}, target_range=[{stopper.p_lo:.4f}, {stopper.p_hi:.4f}], "
            f"observed_target_range=[{stopper.q_lo:.4f}, {stopper.q_hi:.4f}], "
            f"risk={cfg['risk_assessment']} (type1={alpha:.2f}, type2={beta:.2f}), "
            f"judge_fnr={cfg['judge_false_negative_rate']:.4f}, "
            f"judge_fpr={cfg['judge_false_positive_rate']:.4f}, "
            f"prune_ratio={cfg['prune_ratio']}, sample_budget={sample_budget}, "
            f"min_samples={effective_min_samples}, mode={mode}, "
            f"sampling={cfg['sampling']})."
        )

        results_by_subset: Dict[str, List[Tuple[TaskState, Optional[SampleScore]]]] = defaultdict(list)
        free_samples = self._feed_cached(stopper, context, results_by_subset, is_batch)
        if free_samples:
            logger.info(f'Fed {free_samples} already-scored sample(s) to the test before any inference.')

        if stopper.result().decided:  # cached samples alone may resolve the gate
            processed, skipped = 0, len(context.work_items)
            res = stopper.result()
            logger.info(
                f'Sequential early stopping resolved from {res.n} cached sample(s) before any inference: '
                f'verdict={res.verdict} (decision="{res.decision}", target range '
                f'[{stopper.p_lo:.4f}, {stopper.p_hi:.4f}]).'
            )
        elif is_batch:
            processed, skipped = self._run_windowed(stopper, context, results_by_subset, cfg)
        else:
            processed, skipped = self._run_streaming(stopper, context, results_by_subset, cfg)

        # "Early" means work was actually pruned — not merely that a verdict was
        # reached (a decision on the final sample, or max_samples == available
        # work, skips nothing).
        stopped_early = skipped > 0
        final = stopper.result()
        self._stop_info = {
            'stopped_early': stopped_early,
            'method': cfg['strategy'],
            'strategy': cfg['strategy'],
            'prune_ratio': cfg['prune_ratio'],
            'sample_budget': sample_budget,
            'sampling_strategy': cfg['sampling_strategy'],
            'stratify_by': cfg['stratify_by'],
            'sampling': self._sampling_summary(context.work_items, cfg, sample_budget),
            'risk_assessment': cfg['risk_assessment'],
            'judge_false_negative_rate': cfg['judge_false_negative_rate'],
            'judge_false_positive_rate': cfg['judge_false_positive_rate'],
            'fresh_predictions': processed,
            'free_samples': free_samples,
            'samples_skipped': skipped,
            'samples_scored': final.n,
            'samples_total': len(context.work_items) + context.total_cached,
            'token_usage': self._finalize_token_usage(),
            **final.to_dict(),
        }
        if not stopped_early:
            logger.info(
                f'Ran all {final.n} sample(s) (no pruning); verdict={final.verdict} '
                f'(decision={final.decision}, by {final.decided_by}).'
            )
        return results_by_subset

    def _on_error(self, item: _WorkItem, exc: Exception) -> None:
        tb_str = traceback.format_exc()
        logger.error(f'Processing item in subset={item.subset!r} failed: {exc}\nTraceback:\n{tb_str}')
        if self.task_config.ignore_errors:
            logger.warning('Error ignored, continuing with next sample.')
            return
        raise exc

    def _feed_cached(self, stopper, context, results_by_subset, is_batch) -> int:
        """Feed scores available at zero inference cost; return how many were fed.

        Fully-cached scores are re-added to aggregation via the parent context, so
        only feed them here. Batch "review-pending" task states (prediction cached,
        review not yet run) are reviewed now and stored for aggregation.
        """
        free = 0
        for scores in context.cached_scores_by_subset.values():
            for sc in scores:
                if stopper.result().decided:
                    return free
                if self._feed(stopper, sc):
                    free += 1
        if is_batch:
            for subset, pending in list(context.review_pending_by_subset.items()):
                for ts, sc in self._batch_review(subset, pending):
                    if stopper.result().decided:
                        return free
                    results_by_subset[subset].append((ts, sc))
                    if self._feed(stopper, sc):
                        free += 1
            context.review_pending_by_subset = defaultdict(list)  # consumed; don't re-review
        return free

    def _run_streaming(self, stopper, context, results_by_subset, cfg) -> Tuple[int, int]:
        """Per-sample scoring: stop submitting the instant the test decides.

        Cached-prediction work (``needs_predict == False`` — prediction on disk,
        only the review remains) is consumed **before** any fresh inference, so a
        cached set that already resolves the band never triggers a new model call.
        Only genuinely fresh predictions are counted toward ``fresh_predictions``.
        """
        rng = random.Random(cfg['seed'])
        all_review_only = [w for w in context.work_items if not w.needs_predict]
        all_fresh = [w for w in context.work_items if w.needs_predict]
        review_only = self._order_work_items(all_review_only, cfg, rng)
        fresh = self._order_work_items(all_fresh, cfg, rng)
        budget_remaining = self._remaining_budget(stopper, len(context.work_items))
        review_only = review_only[:budget_remaining]
        budget_remaining -= len(review_only)
        fresh = fresh[:max(0, budget_remaining)]

        # Phase 1: review cached predictions first (no new model calls).
        reused = 0
        if review_only and not stopper.result().decided:
            reused = self._stream(
                review_only, stopper, context, results_by_subset,
                desc=f'Reviewing-cached[{self.benchmark_name}]', initial=context.total_cached,
            )
        # Phase 2: fresh predictions (the cost we save on early stop).
        fresh_predictions = 0
        if fresh and not stopper.result().decided:
            fresh_predictions = self._stream(
                fresh, stopper, context, results_by_subset,
                desc=f'Evaluating[{self.benchmark_name}]', initial=context.total_cached + reused,
            )

        trimmed = (len(all_review_only) - len(review_only)) + (len(all_fresh) - len(fresh))
        skipped = trimmed + (len(review_only) - reused) + (len(fresh) - fresh_predictions)
        if skipped > 0:
            res = stopper.result()
            logger.info(
                f'Sequential early stopping resolved: verdict={res.verdict} '
                f'(decision="{res.decision}", by {res.decided_by}) after {res.n} samples; '
                f'ran {fresh_predictions} fresh prediction(s), skipped {skipped} work item(s).'
            )
        return fresh_predictions, skipped

    def _stream(self, items, stopper, context, results_by_subset, *, desc, initial) -> int:
        """Stream ``items`` through the rolling-window pool, feeding the stopper
        after each result and halting submission once it decides. Returns the
        number of items actually processed (in-flight overshoot included)."""
        log_cadence = max(self.task_config.eval_batch_size, 16)
        processed = 0

        def worker(item: _WorkItem) -> Tuple[TaskState, Optional[SampleScore]]:
            return self._process_work_item(item, context.model_prediction_dir)

        def on_result(item: _WorkItem, result: Tuple[TaskState, Optional[SampleScore]]) -> None:
            nonlocal processed
            self._persist_result(item, *result)
            results_by_subset[item.subset].append(result)
            self._feed(stopper, result[1])
            processed += 1
            snap = stopper.result()
            if not snap.decided and snap.n and snap.n % log_cadence == 0:
                logger.info(
                    f'After {snap.n} samples: mean={snap.mean:.4f}, '
                    f'CI=[{snap.ci_lower:.4f}, {snap.ci_upper:.4f}], '
                    f'SPRT_llr={snap.sprt_llr:.3f} -> decision={snap.decision}'
                )

        run_in_threads_with_progress(
            items, worker,
            desc=desc,
            max_workers=self.task_config.eval_batch_size,
            log_interval=HEARTBEAT_INTERVAL_SEC,
            on_result=on_result, on_error=self._on_error, skip_failed=True,
            initial=initial, total=context.grand_total,
            should_stop=lambda: stopper.result().decided,
        )
        return processed

    def _run_windowed(self, stopper, context, results_by_subset, cfg) -> Tuple[int, int]:
        """Batch-scoring benchmarks only: review is deferred and computed in
        windows of ``eval_batch_size``, so we cannot score per result. Predict one
        ``eval_batch_size`` window, batch-review it, feed the test, then check —
        the most responsive cadence possible without per-sample scores.
        """
        all_work_items = list(context.work_items)
        work_items = self._order_work_items(all_work_items, cfg, random.Random(cfg['seed']))
        work_items = work_items[:self._remaining_budget(stopper, len(work_items))]
        window = max(1, self.task_config.eval_batch_size)
        processed = 0
        stopped = False

        for start in range(0, len(work_items), window):
            if stopped:
                break
            round_items = work_items[start:start + window]
            round_states: Dict[str, List[TaskState]] = defaultdict(list)

            def worker(item: _WorkItem) -> Tuple[TaskState, Optional[SampleScore]]:
                return self._process_work_item(item, context.model_prediction_dir)

            def on_result(item: _WorkItem, result: Tuple[TaskState, Optional[SampleScore]]) -> None:
                self._persist_result(item, *result)
                round_states[item.subset].append(result[0])  # review deferred to window end

            run_in_threads_with_progress(
                round_items, worker,
                desc=f'Evaluating[{self.benchmark_name}] round {start // window + 1}',
                max_workers=self.task_config.eval_batch_size,
                log_interval=HEARTBEAT_INTERVAL_SEC,
                on_result=on_result, on_error=self._on_error, skip_failed=True,
                initial=context.total_cached + processed, total=context.grand_total,
            )

            for subset, task_states in round_states.items():
                for ts, sc in self._batch_review(subset, task_states):
                    results_by_subset[subset].append((ts, sc))
                    self._feed(stopper, sc)

            processed += len(round_items)
            res = stopper.result()
            logger.info(
                f'After {res.n} samples: mean={res.mean:.4f}, '
                f'CI=[{res.ci_lower:.4f}, {res.ci_upper:.4f}], SPRT_llr={res.sprt_llr:.3f} '
                f'-> decision={res.decision}'
            )
            if res.decided:
                logger.info(
                    f'Sequential early stopping resolved: verdict={res.verdict} '
                    f'(decision="{res.decision}", by {res.decided_by}) after {res.n} samples; '
                    f'skipping remaining {len(work_items) - processed} predictions.'
                )
                stopped = True

        return processed, len(all_work_items) - processed  # (fresh_predictions, skipped)

    def _order_work_items(self, items: List[_WorkItem], cfg: dict, rng: random.Random) -> List[_WorkItem]:
        """Return the sequential evaluation order for the configured sampling strategy."""
        work_items = list(items)
        if not work_items:
            return []
        if cfg['sampling']['strategy'] == 'uniform':
            rng.shuffle(work_items)
            return work_items
        return self._stratified_order(work_items, cfg, rng)

    def _stratified_order(self, items: List[_WorkItem], cfg: dict, rng: random.Random) -> List[_WorkItem]:
        """Interleave strata so every prefix is close to the full-pool proportions."""
        grouped: Dict[str, List[_WorkItem]] = defaultdict(list)
        for item in items:
            grouped[self._item_stratum(item, cfg)].append(item)
        for group_items in grouped.values():
            rng.shuffle(group_items)

        strata = list(grouped.keys())
        rng.shuffle(strata)
        original_counts = {stratum: len(grouped[stratum]) for stratum in strata}
        selected_counts = {stratum: 0 for stratum in strata}
        total = len(items)
        ordered: List[_WorkItem] = []

        for _ in range(cfg['sampling']['min_per_stratum']):
            for stratum in strata:
                if not grouped[stratum]:
                    continue
                ordered.append(grouped[stratum].pop())
                selected_counts[stratum] += 1

        while len(ordered) < total:
            next_index = len(ordered) + 1
            candidates = [stratum for stratum in strata if grouped[stratum]]
            stratum = max(
                candidates,
                key=lambda s: ((next_index * original_counts[s] / total) - selected_counts[s], original_counts[s]),
            )
            ordered.append(grouped[stratum].pop())
            selected_counts[stratum] += 1
        return ordered

    def _item_stratum(self, item: _WorkItem, cfg: dict) -> str:
        """Resolve a work item's sampling stratum from existing EvalScope segment metadata."""
        values = self._item_segment_values(item, cfg)
        parts = [
            f'{dimension}={self._format_segment_value(values[dimension])}'
            for dimension in cfg['sampling']['dimensions']
        ]
        return '|'.join(parts)

    def _item_segment_values(self, item: _WorkItem, cfg: dict) -> Dict[str, Any]:
        """Resolve all configured dimension values for one work item."""
        sample = item.sample
        task_state = item.task_state
        if sample is None and task_state is not None:
            sample = getattr(task_state, '_sample', None)
        metadata = {}
        if sample is not None:
            metadata.update(getattr(sample, 'metadata', {}) or {})
        if task_state is not None:
            metadata.update(getattr(task_state, 'metadata', {}) or {})

        adapter_segments = self.benchmark.get_sampling_segments(sample=sample, subset=item.subset)
        values = {}
        for dimension in cfg['sampling']['dimensions']:
            if dimension.startswith('metadata.'):
                key = dimension.split('.', 1)[1]
                values[dimension] = metadata.get(key, 'unknown')
            else:
                values[dimension] = adapter_segments.get(dimension, 'unknown')
        return values

    def _format_segment_value(self, value: Any) -> str:
        """Format a segment value into a stable, readable stratum component."""
        if isinstance(value, (list, tuple)):
            return '/'.join(str(part) for part in value)
        if value is None:
            return 'unknown'
        return str(value)

    def _sampling_summary(self, items: List[_WorkItem], cfg: dict, sample_budget: int) -> dict:
        """Summarize configured sampling for report metadata and reproducibility."""
        ordered = self._order_work_items(items, cfg, random.Random(cfg['seed']))
        preview = ordered[:min(sample_budget, len(ordered))]
        strata = defaultdict(int)
        budget_strata = defaultdict(int)
        for item in items:
            strata[self._item_stratum(item, cfg)] += 1
        for item in preview:
            budget_strata[self._item_stratum(item, cfg)] += 1
        return {
            **cfg['sampling'],
            'strata': dict(sorted(strata.items())),
            'budget_strata': dict(sorted(budget_strata.items())),
        }

    @staticmethod
    def _remaining_budget(stopper: SequentialStopper, fallback: int) -> int:
        """Return how many more work items may be submitted under the sample budget."""
        if stopper.max_samples is None:
            return fallback
        return max(0, min(fallback, stopper.max_samples - stopper.result().n))

    def _batch_review(self, subset: str, task_states: List[TaskState]) -> List[Tuple[TaskState, SampleScore]]:
        """Batch-review a chunk of task states; pair each score with its own task state.

        ``review_subset`` drops samples whose preliminary score is ``None`` and
        returns only the surviving scores (in input order), so a positional
        ``zip`` against ``task_states`` would misalign every score after a dropped
        one. The survivors are an in-order subsequence of the inputs, so we
        re-align by advancing through ``task_states`` and matching on ``sample_id``.
        """
        if not task_states:
            return []
        scores = self.batch_reviewer.review_subset(subset, task_states, review_fn=self._review_task_state)
        pairs: List[Tuple[TaskState, SampleScore]] = []
        ts_iter = iter(task_states)
        for sc in scores:
            if sc is None:
                continue
            for ts in ts_iter:  # skip task states whose preliminary score was dropped
                if ts.sample_id == sc.sample_id:
                    pairs.append((ts, sc))
                    break
        return pairs

    def _add_token_usage(self, task_state: TaskState, source: str) -> None:
        """Accumulate token usage for work performed by this sequential run."""
        self._ensure_token_usage()
        perfs = [m.perf_metrics for m in task_state.messages
                 if m.role == 'assistant' and m.perf_metrics is not None]
        if not perfs and task_state.output.perf_metrics is not None:
            perfs = [task_state.output.perf_metrics]
        bucket = self._sequential_token_usage.setdefault(source, {
            'requests': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
        })
        for perf in perfs:
            input_tokens = int(getattr(perf, 'input_tokens', 0) or 0)
            output_tokens = int(getattr(perf, 'output_tokens', 0) or 0)
            bucket['requests'] += 1
            bucket['input_tokens'] += input_tokens
            bucket['output_tokens'] += output_tokens
            bucket['total_tokens'] += input_tokens + output_tokens

    def _finalize_token_usage(self) -> dict:
        """Return token totals and a short note about what is counted."""
        self._ensure_token_usage()
        fresh = self._sequential_token_usage.get('fresh_predictions', {})
        return {
            'consumed': dict(fresh),
            'fresh_predictions': dict(fresh),
            'counted_from': 'perf_metrics for fresh predictions processed in this run',
        }

    def _ensure_token_usage(self) -> None:
        """Initialize token accounting for objects constructed in tests via ``__new__``."""
        if hasattr(self, '_sequential_token_usage'):
            return
        self._sequential_token_usage = {
            'fresh_predictions': {
                'requests': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
            },
        }

    def _persist_result(
        self,
        item: _WorkItem,
        task_state: TaskState,
        sample_score: Optional[SampleScore],
    ) -> None:
        """Persist a result and accumulate sequential-run token totals."""
        super()._persist_result(item=item, task_state=task_state, sample_score=sample_score)
        if item.needs_predict:
            self._add_token_usage(task_state, 'fresh_predictions')

    def _feed(self, stopper: SequentialStopper, sample_score: Optional[SampleScore]) -> bool:
        """Feed a score's accuracy to the test; return True if it was consumed."""
        acc = self._extract_accuracy(sample_score)
        if acc is None:
            return False
        stopper.update(acc)
        return True

    def _aggregate_scores(self, dataset_dict, context, results_by_subset):
        """Aggregate from the scores collected during the rounds.

        The sequential pool already produced final per-sample scores for both
        per-sample and batch-scoring benchmarks (the latter reviewed at each
        round boundary), so — unlike the parent — no second batch-review pass is
        run here, avoiding double scoring cost.
        """
        agg_score_dict = {}
        for subset in dataset_dict:
            cached_scores = context.cached_scores_by_subset.get(subset, [])
            pool_scores = [sc for _, sc in results_by_subset.get(subset, []) if sc is not None]
            all_scores = cached_scores + pool_scores
            if not all_scores:
                logger.info(f'No valid scores generated for subset: {subset}, skipping.')
                continue
            logger.info(f'Aggregating scores for subset: {subset}')
            agg_score_dict[subset] = self.benchmark.aggregate_scores(sample_scores=all_scores)
        return agg_score_dict

    def _extract_accuracy(self, sample_score: Optional[SampleScore]) -> Optional[float]:
        """Pull a scalar accuracy in [0, 1] from a sample score, if available.

        Values outside [0, 1] are clamped (the band semantics assume an
        accuracy-like metric); a warning is emitted once. Scores from which no
        scalar can be extracted are ignored (also warned once), which stalls
        progress — surfaced so the user can pick a benchmark with an ``acc``
        metric or drop early stopping.
        """
        if sample_score is None:
            raw = None
        else:
            value = sample_score.score.value
            raw = value.get('acc') if isinstance(value, dict) and 'acc' in value else sample_score.score.main_value

        if raw is None:
            if not self._warned_no_score:
                self._warned_no_score = True
                logger.warning(
                    f"Could not extract a scalar score from a sample in '{self.benchmark_name}'; "
                    'such samples are ignored by the sequential test (no early-stop progress). '
                    'Early stopping expects an accuracy-like metric (e.g. `acc`).'
                )
            return None
        try:
            acc = float(raw)
        except (TypeError, ValueError):
            return None
        if (acc < 0.0 or acc > 1.0) and not self._warned_out_of_range:
            self._warned_out_of_range = True
            logger.warning(
                f"Sample score {acc:.4f} in '{self.benchmark_name}' is outside [0, 1]; clamping. "
                'The sequential early-stopping band assumes an accuracy-like metric in [0, 1].'
            )
        return min(1.0, max(0.0, acc))

    def get_report(self, agg_score_dict):
        """Generate the report and persist sequential-stop diagnostics."""
        report = super().get_report(agg_score_dict)
        stop_info = getattr(self, '_stop_info', None)
        if stop_info is not None:
            metadata = report.metadata or {}
            metadata['sequential_stop'] = stop_info
            report.metadata = metadata
            report.to_json(self.cache_manager.get_report_file())

            sidecar = os.path.join(
                self.cache_manager.get_report_path(), f'{self.benchmark_name}_sequential_stop.jsonl'
            )
            try:
                with open(sidecar, 'w', encoding='utf-8') as f:
                    json.dump(stop_info, f, ensure_ascii=False)
                    f.write('\n')
                logger.info(
                    f'Sequential stop decision embedded in report metadata and written to: {sidecar}'
                )
            except OSError as e:
                logger.warning(f'Failed to write sequential stop sidecar: {e}')
        return report
