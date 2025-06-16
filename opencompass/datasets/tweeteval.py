from opencompass.registry import LOAD_DATASET
from datasets import Dataset, load_dataset

from .base import BaseDataset

@LOAD_DATASET.register_module()
class TweetEvalSentimentDataset(BaseDataset):

    @staticmethod
    def load(path=None):
        # Load the TweetEval sentiment task
        print("Try Loading the dataset...")
        dataset = load_dataset("tweet_eval", "sentiment", split="validation")
        data = []
        label_mapping = {0:"A",1:"B",2:"C"}
        for row in dataset:
            data.append({
                "text": row["text"],
                "label": label_mapping.get(row["label"],row["label"])
            })
        return Dataset.from_list(data)