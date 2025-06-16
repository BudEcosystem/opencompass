from mmengine.config import read_base
from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # Import HealthBench datasets
    from opencompass.configs.datasets.healthbench.healthbench_gen import \
        healthbench_datasets, healthbench_demo_datasets

# API meta template for Qwen3-4B
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# Qwen3-4B model configuration
models = [
    dict(
        abbr='qwen3-4b',
        type=OpenAI,
        path='qwen3-4b',
        key='ENV',  # The key will be obtained from $OPENAI_API_KEY
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        tokenizer_path='Qwen/Qwen3-4B',
        batch_size=8
    ),
]

# Use demo dataset for quick testing (100 examples)
# For full evaluation, use: datasets = healthbench_datasets
datasets = healthbench_demo_datasets

# Inference configuration
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)
    ),
)

# You can also evaluate specific subsets:
# datasets = [d for d in healthbench_datasets if d['abbr'] == 'healthbench_hard']
# 
# Or use full datasets:
# datasets = healthbench_datasets
#
# Or use LLM-graded evaluation (requires EVAL_OPENAI_API_KEY):
# from opencompass.configs.datasets.healthbench.healthbench_gen import healthbench_demo_datasets_with_llm
# datasets = healthbench_demo_datasets_with_llm