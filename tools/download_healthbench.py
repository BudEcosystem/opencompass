#!/usr/bin/env python3
"""
Script to download and prepare HealthBench dataset for OpenCompass evaluation.

Usage:
    python tools/download_healthbench.py --data-dir ./data/healthbench
"""

import argparse
import json
import os
import requests
from typing import Dict, List


# HealthBench dataset URLs from OpenAI's simple-evals
HEALTHBENCH_URLS = {
    'main': 'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl',
    'hard': 'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl',
    'consensus': 'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl'
}


def download_file(url: str, output_path: str):
    """Download a file from URL to output path."""
    print(f"Downloading {url} to {output_path}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write file in chunks
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print(f"Downloaded successfully: {output_path}")


def validate_jsonl(filepath: str) -> int:
    """Validate JSONL file and return number of examples."""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    return -1
    return count


def main():
    parser = argparse.ArgumentParser(description='Download HealthBench dataset for OpenCompass')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/healthbench',
        help='Directory to save the dataset (default: ./data/healthbench)'
    )
    parser.add_argument(
        '--subsets',
        type=str,
        nargs='+',
        default=['main', 'hard', 'consensus'],
        choices=['main', 'hard', 'consensus'],
        help='Which subsets to download (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    
    args = parser.parse_args()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Download each subset
    for subset in args.subsets:
        filename = f'healthbench_{subset}.jsonl'
        filepath = os.path.join(args.data_dir, filename)
        
        # Check if file exists
        if os.path.exists(filepath) and not args.force:
            print(f"File already exists: {filepath}")
            # Validate existing file
            num_examples = validate_jsonl(filepath)
            if num_examples > 0:
                print(f"  Contains {num_examples} examples")
                continue
            else:
                print(f"  File is corrupted, re-downloading...")
        
        # Download file
        url = HEALTHBENCH_URLS[subset]
        try:
            download_file(url, filepath)
            
            # Validate downloaded file
            num_examples = validate_jsonl(filepath)
            if num_examples > 0:
                print(f"  Successfully downloaded {num_examples} examples")
            else:
                print(f"  Error: Downloaded file is corrupted")
                os.remove(filepath)
        except Exception as e:
            print(f"Error downloading {subset}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
    
    print("\nDownload complete!")
    print(f"Dataset files are saved in: {os.path.abspath(args.data_dir)}")
    print("\nTo use HealthBench in evaluation, add this to your config:")
    print("```python")
    print("from opencompass.configs.datasets.healthbench.healthbench_gen import healthbench_datasets")
    print("datasets = healthbench_datasets")
    print("```")


if __name__ == '__main__':
    main()