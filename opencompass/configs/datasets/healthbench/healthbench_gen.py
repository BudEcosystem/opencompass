from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HealthBenchDataset, HealthBenchEvaluator

# Reader configuration
healthbench_reader_cfg = dict(
    input_columns=['conversation_history', 'question'],
    output_column='rubric_items',  # Use rubric_items as placeholder for evaluation
    train_split='test',  # Use test split for both train and test
    test_split='test',  # HealthBench only has test split
)

# Inference configuration
healthbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{conversation_history}\n\nUser: {question}\nAssistant:'
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

# Evaluation configuration
healthbench_eval_cfg = dict(
    evaluator=dict(
        type=HealthBenchEvaluator,
        use_grading_model=False,  # Set to True to enable LLM grading
        grading_model_name='gpt-4',  # Model to use for grading
        # API configuration (optional - will use environment variables if not set)
        # openai_api_base='https://api.openai.com/v1/chat/completions',
        # openai_api_key='your-api-key-here',  # Can also be a list of keys
        temperature=0.0,  # Low temperature for consistent grading
        max_out_len=100,  # Short responses for YES/NO answers
    ),
)

# Configuration with LLM grading enabled (requires API access)
healthbench_eval_cfg_with_llm = dict(
    evaluator=dict(
        type=HealthBenchEvaluator,
        use_grading_model=True,
        grading_model_name='qwen3-32b',
        # API settings will be read from environment variables:
        # EVAL_OPENAI_API_KEY and EVAL_OPENAI_API_BASE
        temperature=0.0,
        max_out_len=100, 
    ),
)

# Dataset configurations for different subsets
healthbench_datasets = [
    dict(
        abbr='healthbench_main',
        type=HealthBenchDataset,
        path='opencompass/healthbench',
        subset='main',
        num_examples=None,  # Use all examples
        reader_cfg=healthbench_reader_cfg,
        infer_cfg=healthbench_infer_cfg,
        eval_cfg=healthbench_eval_cfg,
    ),
    dict(
        abbr='healthbench_hard',
        type=HealthBenchDataset,
        path='opencompass/healthbench',
        subset='hard',
        num_examples=None,
        reader_cfg=healthbench_reader_cfg,
        infer_cfg=healthbench_infer_cfg,
        eval_cfg=healthbench_eval_cfg,
    ),
    dict(
        abbr='healthbench_consensus',
        type=HealthBenchDataset,
        path='opencompass/healthbench',
        subset='consensus',
        num_examples=None,
        reader_cfg=healthbench_reader_cfg,
        infer_cfg=healthbench_infer_cfg,
        eval_cfg=healthbench_eval_cfg,
    ),
]

# Configuration for main subset only (commonly used)
healthbench_main_datasets = [
    dict(
        abbr='healthbench',
        type=HealthBenchDataset,
        path='opencompass/healthbench',
        subset='main',
        num_examples=None,
        reader_cfg=healthbench_reader_cfg,
        infer_cfg=healthbench_infer_cfg,
        eval_cfg=healthbench_eval_cfg,
    ),
]

# Configuration for demo/testing with limited examples
healthbench_demo_datasets = [
    dict(
        abbr='healthbench_demo',
        type=HealthBenchDataset,
        path='opencompass/healthbench',
        subset='main',
        num_examples=100,  # Only use 100 examples for quick testing
        seed=42,
        reader_cfg=healthbench_reader_cfg,
        infer_cfg=healthbench_infer_cfg,
        eval_cfg=healthbench_eval_cfg,
    ),
]

# Configuration with LLM grading for all subsets
healthbench_datasets_with_llm = [
    dict(
        abbr='healthbench_main_llm',
        type=HealthBenchDataset,
        path='opencompass/healthbench',
        subset='main',
        num_examples=None,
        reader_cfg=healthbench_reader_cfg,
        infer_cfg=healthbench_infer_cfg,
        eval_cfg=healthbench_eval_cfg_with_llm,
    ),
    dict(
        abbr='healthbench_hard_llm',
        type=HealthBenchDataset,
        path='opencompass/healthbench',
        subset='hard',
        num_examples=None,
        reader_cfg=healthbench_reader_cfg,
        infer_cfg=healthbench_infer_cfg,
        eval_cfg=healthbench_eval_cfg_with_llm,
    ),
    dict(
        abbr='healthbench_consensus_llm',
        type=HealthBenchDataset,
        path='opencompass/healthbench',
        subset='consensus',
        num_examples=None,
        reader_cfg=healthbench_reader_cfg,
        infer_cfg=healthbench_infer_cfg,
        eval_cfg=healthbench_eval_cfg_with_llm,
    ),
]

# Demo configuration with LLM grading
healthbench_demo_datasets_with_llm = [
    dict(
        abbr='healthbench_demo_llm',
        type=HealthBenchDataset,
        path='opencompass/healthbench',
        subset='main',
        num_examples=10,  # Very small sample for testing LLM grading
        seed=42,
        reader_cfg=healthbench_reader_cfg,
        infer_cfg=healthbench_infer_cfg,
        eval_cfg=healthbench_eval_cfg_with_llm,
    ),
]