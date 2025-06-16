from mmengine.config import read_base

from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # choose a list of datasets
    # from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets as datasets
    # and output the results in a choosen format
    # from opencompass.configs.summarizers.medium import summarizer
    from opencompass.configs.datasets.tweeteval.tweeteval_gen import tweeteval_datasets as datasets

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='qwen3-4b',
        type=OpenAI,
        path='qwen3-4b',
        key=
        'ENV',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        tokenizer_path='Qwen/Qwen3-4B',
        batch_size=8),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)