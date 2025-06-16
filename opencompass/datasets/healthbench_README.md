# HealthBench Dataset Integration

This directory contains the OpenCompass integration for OpenAI's HealthBench dataset, which evaluates large language models in realistic healthcare scenarios.

## Overview

HealthBench is a comprehensive evaluation framework developed by OpenAI in collaboration with 262 physicians across 60 countries and 26 medical specialties. It features:

- 5,000 multi-turn conversations between models and users/healthcare professionals
- Physician-written rubric criteria for evaluation
- Three dataset variants: main, hard, and consensus
- Seven evaluation themes: emergency referrals, global health, health data tasks, context-seeking, expertise-tailored communication, response depth, and responding under uncertainty

## Setup

### 1. Download the Dataset

First, download the HealthBench dataset files:

```bash
# Download all subsets
python tools/download_healthbench.py --data-dir ./data/healthbench

# Download specific subsets
python tools/download_healthbench.py --data-dir ./data/healthbench --subsets main hard

# Force re-download
python tools/download_healthbench.py --data-dir ./data/healthbench --force
```

The script will download the dataset files to `./data/healthbench/`:
- `healthbench_main.jsonl` - Main dataset (full evaluation)
- `healthbench_hard.jsonl` - Challenging subset
- `healthbench_consensus.jsonl` - Physician consensus criteria subset

### 2. Run Evaluation

Use the provided configuration to evaluate models:

```bash
# Quick demo evaluation (100 examples)
python run.py examples/eval_healthbench.py

# Full evaluation on main dataset
python run.py examples/eval_healthbench.py --datasets healthbench_datasets

# Evaluate specific subset
python run.py examples/eval_healthbench.py --datasets healthbench_hard
```

## Configuration

The dataset supports several configuration options:

```python
from opencompass.configs.datasets.healthbench.healthbench_gen import healthbench_datasets

# Configure specific subset with limited examples
healthbench_config = dict(
    abbr='healthbench_custom',
    type=HealthBenchDataset,
    path='opencompass/healthbench',
    subset='main',  # 'main', 'hard', or 'consensus'
    num_examples=500,  # Limit number of examples
    seed=42,  # Random seed for sampling
    include_physician_completion=False,  # Include reference answers
    reader_cfg=dict(
        input_columns=['conversation_history', 'question'],
        output_column='reference',
    ),
    infer_cfg=dict(...),
    eval_cfg=dict(
        evaluator=dict(
            type=HealthBenchEvaluator,
            use_grading_model=False,  # Enable LLM-based grading
        ),
    ),
)
```

## Evaluation Metrics

The HealthBenchEvaluator scores model responses against physician-written rubric criteria:

- **Overall Score**: Percentage of rubric points earned across all examples
- **Tag Scores**: Performance breakdown by example tags (e.g., emergency care, uncertainty handling)
- **Rubric Details**: Individual criterion evaluation results

### Scoring Method

1. Each example has multiple rubric items with associated point values
2. Model responses are evaluated against each criterion
3. Final score = (earned points / total points) × 100

### Grading Options

- **Rule-based**: Simple keyword matching (default)
- **LLM-based**: Use GPT-4 or similar for criterion evaluation

#### Configuring LLM-based Grading

To enable LLM-based grading, you need to configure API access:

1. **Using Environment Variables** (recommended):
   ```bash
   export EVAL_OPENAI_API_KEY="your-api-key"
   export EVAL_OPENAI_API_BASE="https://api.openai.com/v1/"  # or your custom endpoint
   ```
   
   Note: The evaluator uses `EVAL_OPENAI_API_KEY` and `EVAL_OPENAI_API_BASE` to avoid conflicts with model evaluation API keys.

2. **In Configuration**:
   ```python
   healthbench_eval_cfg = dict(
       evaluator=dict(
           type=HealthBenchEvaluator,
           use_grading_model=True,
           grading_model_name='gpt-4',  # or 'gpt-3.5-turbo'
           openai_api_base='https://api.openai.com/v1/chat/completions',
           openai_api_key='your-api-key',  # Can be a list for round-robin
           temperature=0.0,  # Low for consistent grading
           max_out_len=100,
       ),
   )
   ```

3. **Using Pre-configured Settings**:
   ```python
   from opencompass.configs.datasets.healthbench.healthbench_gen import healthbench_datasets_with_llm
   
   # This uses LLM grading with settings from environment variables
   datasets = healthbench_datasets_with_llm
   ```

## Dataset Structure

Each example contains:
- `example_id`: Unique identifier
- `messages`: Conversation history (user/assistant turns)
- `rubric_items`: List of evaluation criteria with point values
- `example_tags`: Categories for performance analysis
- `topic`, `conversation_type`, `difficulty`: Metadata

## Implementation Details

### Files

- `opencompass/datasets/healthbench.py` - Dataset loader and evaluator
- `opencompass/configs/datasets/healthbench/healthbench_gen.py` - Configurations
- `tools/download_healthbench.py` - Dataset download script
- `examples/eval_healthbench.py` - Example evaluation script

### Integration Points

1. **Dataset Registration**: `@LOAD_DATASET.register_module()`
2. **Evaluator Registration**: `@ICL_EVALUATORS.register_module()`
3. **Data Path Mapping**: Added to `datasets_info.py`
4. **Module Import**: Added to `datasets/__init__.py`

## Limitations

1. **Grading Accuracy**: While LLM-based grading is implemented, the rule-based fallback uses simple keyword matching which may not capture all nuances.
2. **Dataset URLs**: The download URLs point to OpenAI's blob storage and may change. Consider hosting locally for production use.
3. **Evaluation Speed**: Full dataset evaluation can be slow, especially with LLM grading enabled (each rubric item requires an API call).
4. **API Costs**: LLM-based grading can be expensive at scale due to the number of API calls (one per rubric item).

## References

- [HealthBench Paper](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf)
- [OpenAI HealthBench Announcement](https://openai.com/index/healthbench/)
- [Simple-evals Implementation](https://github.com/openai/simple-evals/blob/main/healthbench_eval.py)