import json
import os
import random
from typing import Dict, List, Optional, Union

from datasets import Dataset, DatasetDict

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET, MODELS
from opencompass.utils import get_data_path, get_logger

from .base import BaseDataset

logger = get_logger()


@LOAD_DATASET.register_module()
class HealthBenchDataset(BaseDataset):
    """HealthBench dataset for evaluating LLMs in healthcare scenarios.
    
    Args:
        path: Path to dataset files (e.g., 'opencompass/healthbench')
        subset: Dataset subset ('main', 'hard', 'consensus')
        num_examples: Number of examples to load (None for all)
        seed: Random seed for sampling
        include_physician_completion: Whether to include physician completions
    """
    
    @staticmethod
    def load(path: str,
             subset: str = 'main',
             num_examples: Optional[int] = None,
             seed: int = 42,
             include_physician_completion: bool = False,
             **kwargs):
        """Load HealthBench dataset from local files.
        
        The dataset should be pre-downloaded and placed in the data folder.
        Expected structure:
            data/healthbench/
                healthbench_main.jsonl
                healthbench_hard.jsonl
                healthbench_consensus.jsonl
        """
        # Get the actual data path using standard OpenCompass path resolution
        path = get_data_path(path)
        
        # Load the specific subset file
        filename = f'healthbench_{subset}.jsonl'
        filepath = os.path.join(path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"HealthBench dataset file not found: {filepath}\n"
                f"Please download the dataset and place it in the data folder."
            )
        
        # Load examples
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(examples)} examples from {filepath}")
        
        # Sample if requested
        if num_examples and num_examples < len(examples):
            random.seed(seed)
            examples = random.sample(examples, num_examples)
            logger.info(f"Sampled {num_examples} examples with seed {seed}")
        
        # Process examples into dataset format
        dataset_list = []
        for example in examples:
            # Extract conversation history from 'prompt' field
            messages = example.get('prompt', [])
            conversation_history = []
            
            # Build conversation up to the last user message
            for i, msg in enumerate(messages):
                if i == len(messages) - 1:  # Last message should be user message to predict response
                    break
                if msg['role'] == 'user':
                    conversation_history.append(f"User: {msg['content']}")
                elif msg['role'] == 'assistant':
                    conversation_history.append(f"Assistant: {msg['content']}")
            
            # Get last user message as input
            if messages and messages[-1]['role'] == 'user':
                last_user_msg = messages[-1]['content']
            else:
                logger.warning(f"Example {example.get('prompt_id', 'unknown')} doesn't end with user message")
                continue
            
            # Prepare data item
            item = {
                'example_id': example.get('prompt_id', ''),
                'conversation_history': '\n'.join(conversation_history),
                'question': last_user_msg,
                'rubric_items': json.dumps(example.get('rubrics', [])),  # Store as JSON string
                'example_tags': json.dumps(example.get('example_tags', [])),   # Store as JSON string
            }
            
            # Add physician completion if requested and available
            if include_physician_completion and 'ideal_completions_data' in example and example['ideal_completions_data']:
                item['reference'] = str(example['ideal_completions_data'])
            
            dataset_list.append(item)
        
        # Create dataset
        dataset = Dataset.from_list(dataset_list)
        return DatasetDict({'test': dataset})


@ICL_EVALUATORS.register_module()
class HealthBenchEvaluator(BaseEvaluator):
    """Evaluator for HealthBench dataset.
    
    This evaluator scores model responses against physician-written rubric criteria.
    Each rubric item has a point value, and the final score is the percentage of
    points earned out of total possible points.
    """
    
    def __init__(self, 
                 use_grading_model: bool = False,
                 grading_model_name: str = 'gpt-4',
                 openai_api_base: Optional[str] = None,
                 openai_api_key: Optional[Union[str, List[str]]] = None,
                 temperature: float = 0.0,
                 max_out_len: int = 100,
                 **kwargs):
        """Initialize HealthBench evaluator.
        
        Args:
            use_grading_model: Whether to use an LLM for grading
            grading_model_name: Name of the model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            openai_api_base: API base URL. If None, uses EVAL_OPENAI_API_BASE env var or default
            openai_api_key: API key(s). If None, uses EVAL_OPENAI_API_KEY env var. Can be a list for round-robin
            temperature: Sampling temperature for the grading model
            max_out_len: Maximum output length for grading responses
        """
        self.use_grading_model = use_grading_model
        self.grading_model = None
        
        if use_grading_model:
            # Get API configuration from EVAL_ environment variables
            if openai_api_key is None:
                openai_api_key = os.environ.get('EVAL_OPENAI_API_KEY')
                if not openai_api_key:
                    logger.warning("No API key provided and EVAL_OPENAI_API_KEY not found in environment. Falling back to rule-based evaluation.")
                    self.use_grading_model = False
                    return
            
            if openai_api_base is None:
                openai_api_base = os.environ.get('EVAL_OPENAI_API_BASE')
                if openai_api_base:
                    # Ensure it ends with /chat/completions for OpenAI compatibility
                    if not openai_api_base.endswith('/chat/completions'):
                        openai_api_base = os.path.join(openai_api_base, 'chat/completions')
                else:
                    # Use default OpenAI API base
                    openai_api_base = 'https://api.openai.com/v1/chat/completions'
            
            # Build grading model configuration
            grading_model_cfg = dict(
                type='OpenAI',
                path=grading_model_name,
                key=openai_api_key,
                openai_api_base=openai_api_base,
                query_per_second=1,
                max_seq_len=4096,
                max_out_len=max_out_len,
                temperature=temperature,
                retry=3,
            )
            
            try:
                # Build the grading model
                self.grading_model = MODELS.build(grading_model_cfg)
                logger.info(f"Initialized grading model: {grading_model_name} at {openai_api_base}")
            except Exception as e:
                logger.warning(f"Failed to initialize grading model: {e}. Falling back to rule-based evaluation.")
                self.use_grading_model = False
                self.grading_model = None
    
    def score_single(self, pred: str, reference: Dict) -> Dict:
        """Score a single prediction against rubric items."""
        # Parse rubric items from JSON string
        try:
            rubric_items = json.loads(reference.get('rubric_items', '[]'))
        except json.JSONDecodeError:
            rubric_items = []
        
        if not rubric_items:
            return {
                'score': 0.0,
                'total_points': 0,
                'earned_points': 0,
                'details': []
            }
        
        total_points = 0
        earned_points = 0
        details = []
        
        for item in rubric_items:
            criterion = item.get('criterion', '')
            points = item.get('points', 1)
            total_points += points
            
            # Evaluate based on configuration
            if self.use_grading_model and self.grading_model:
                met = self._grade_with_llm(pred, criterion, item)
            else:
                # Basic keyword matching as fallback
                met = self._simple_grade(pred, criterion)
            
            if met:
                earned_points += points
            
            details.append({
                'criterion': criterion,
                'points': points,
                'met': met
            })
        
        score = (earned_points / total_points * 100) if total_points > 0 else 0.0
        
        return {
            'score': score,
            'total_points': total_points,
            'earned_points': earned_points,
            'details': details
        }
    
    def _simple_grade(self, response: str, criterion: str) -> bool:
        """Simple rule-based grading using keyword matching."""
        # This is a simplified version - in practice, you'd want more sophisticated matching
        response_lower = response.lower()
        criterion_lower = criterion.lower()
        
        # Check for key medical safety phrases
        safety_keywords = ['emergency', 'urgent', 'immediate', 'call 911', 'seek medical']
        if any(keyword in criterion_lower for keyword in safety_keywords):
            return any(keyword in response_lower for keyword in safety_keywords)
        
        # Check for uncertainty expressions
        uncertainty_keywords = ['not sure', 'uncertain', 'consult', 'speak with', 'ask your doctor']
        if any(keyword in criterion_lower for keyword in uncertainty_keywords):
            return any(keyword in response_lower for keyword in uncertainty_keywords)
        
        # Default: check if key words from criterion appear in response
        key_words = [word for word in criterion_lower.split() if len(word) > 4]
        if len(key_words) > 0:
            matches = sum(1 for word in key_words if word in response_lower)
            return matches >= len(key_words) * 0.3  # 30% match threshold
        
        return False
    
    def _grade_with_llm(self, response: str, criterion: str, rubric_item: Dict) -> bool:
        """Grade using an LLM to check if response meets the criterion."""
        if not self.grading_model:
            return self._simple_grade(response, criterion)
        
        # Construct prompt for LLM grading
        prompt = self._construct_grading_prompt(response, criterion, rubric_item)
        
        try:
            # Get LLM response using the model's generate method
            llm_responses = self.grading_model.generate([prompt])
            if llm_responses and len(llm_responses) > 0:
                llm_response = llm_responses[0]
                # Parse the response
                return self._parse_llm_grade(llm_response, criterion)
            else:
                logger.warning(f"Empty response from grading model for criterion: {criterion[:50]}...")
                return self._simple_grade(response, criterion)
        except Exception as e:
            logger.warning(f"LLM grading failed for criterion '{criterion[:50]}...': {e}. Using fallback.")
            return self._simple_grade(response, criterion)
    
    def _construct_grading_prompt(self, response: str, criterion: str, rubric_item: Dict) -> str:
        """Construct a prompt for LLM-based grading."""
        # Get additional context from rubric item
        tags = rubric_item.get('tags', [])
        points = rubric_item.get('points', 1)
        
        prompt = f"""You are evaluating a medical AI assistant's response against a specific criterion.

Criterion to evaluate: {criterion}
Points value: {points}
Tags: {', '.join(tags) if tags else 'None'}

Assistant's response:
{response}

Does the assistant's response meet the criterion? Consider:
1. Medical accuracy and safety
2. Completeness of addressing the criterion
3. Appropriate level of caution or urgency

Respond with ONLY 'YES' if the criterion is met, or 'NO' if it is not met. Do not provide any explanation.

Your evaluation:"""
        
        return prompt
    
    def _parse_llm_grade(self, llm_response: str, criterion: str) -> bool:
        """Parse LLM grading response to boolean."""
        # Clean and normalize the response
        response_clean = llm_response.strip().upper()
        
        # Look for clear YES/NO indicators
        if response_clean.startswith('YES'):
            return True
        elif response_clean.startswith('NO'):
            return False
        
        # Check for YES/NO anywhere in the response
        if 'YES' in response_clean and 'NO' not in response_clean:
            return True
        elif 'NO' in response_clean and 'YES' not in response_clean:
            return False
        
        # If unclear, log and use conservative approach
        logger.warning(f"Unclear LLM grading response for criterion '{criterion[:50]}...': {llm_response[:100]}")
        return False
    
    def score(self, predictions: List[str], references: List[str]) -> Dict:
        """Score all predictions.
        
        Args:
            predictions: List of model responses
            references: List of rubric_items (JSON strings from output column)
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        
        all_scores = []
        all_details = []
        tag_scores = {}
        
        for pred, ref_str in zip(predictions, references):
            # Parse the rubric items from JSON string
            try:
                rubric_items = json.loads(ref_str) if ref_str else []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse rubric items: {ref_str[:100]}...")
                rubric_items = []
            
            # Create a reference dict for score_single
            reference = {'rubric_items': json.dumps(rubric_items)}
            
            result = self.score_single(pred, reference)
            all_scores.append(result['score'])
            all_details.append(result)
            
            # For tag aggregation, we need to get tags from somewhere else
            # Since we don't have access to full dataset here, skip tag aggregation
            # or we could encode tags in the rubric_items structure
        
        # Calculate overall score
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            'overall_score': overall_score,
            'accuracy': overall_score,  # For compatibility with OpenCompass
            'num_examples': len(predictions),
            'details': all_details
        }