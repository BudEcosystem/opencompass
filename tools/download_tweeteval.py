from datasets import load_dataset
dataset = load_dataset('tweet_eval', 'sentiment', split="validation")
dataset.to_json('data/tweeteval_sentiment.jsonl', orient='records', lines=True)
