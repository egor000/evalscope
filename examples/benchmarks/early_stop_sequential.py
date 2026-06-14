"""Generic sequential early-stopping evaluation.

Works with *any* benchmark whose per-sample score exposes a scalar in [0, 1]
(the ``acc`` metric, or the main score clamped to [0, 1]). Add an ``early_stop``
block to a dataset's ``dataset_args`` and samples are evaluated in mini-batch
rounds; a selected sequential strategy checks after each
round whether the model's accuracy lies above, below, or within the target band.
The band margin is derived from target_accuracy, prune_ratio, total dataset size,
and the selected risk_assessment profile. Once resolved, the remaining samples
are skipped.

The stop decision + final confidence sequence are embedded in report metadata
and written to ``<output>/reports/<model>/<dataset>_sequential_stop.jsonl``.
"""
import os

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

# `early_stop` can be attached to each dataset independently (targets differ
# per benchmark). Here we early-stop both gsm8k and live_code_bench.
early_stop = {
    'strategy': 'sprt',            # 'sprt' | 'bayes' | 'agrappa' | 'fixed'
    'target_accuracy': 0.80,       # target bar; required
    'prune_ratio': 0.30,           # evaluate about 30% before budget fallback
    'risk_assessment': 'balance',  # conservative | balance | aggressive
    'min_samples': 30,             # do not stop before this many samples
}

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type=EvalType.OPENAI_API,
    datasets=['gsm8k', 'live_code_bench'],
    dataset_args={
        'gsm8k': {'early_stop': early_stop},
        'live_code_bench': {
            'subset_list': ['release_latest'],
            'early_stop': {**early_stop, 'target_accuracy': 0.60},
        },
    },
    eval_batch_size=8,
    generation_config={'temperature': 0.0},
)

if __name__ == '__main__':
    run_task(task_cfg=task_cfg)
