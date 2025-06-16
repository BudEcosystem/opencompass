from opencompass.datasets import TweetEvalSentimentDataset
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate

tweeteval_sentiment_dataloader = dict(
    dataset=dict(type=TweetEvalSentimentDataset),
    batch_size=1,
    num_workers=0
)
# tweeteval_sentiment_evaluator = dict(
#     type='ClassificationEvaluator',
#     metric='accuracy'
# )

tweeteval_sentiment_evaluator = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABC'))

# tweet_eval_prompt = dict(
#     type='PromptTemplate',
#     template='What is the sentiment of the following tweet?\nTweet: "{text}"\nAnswer:',
#     choices=['negative', 'neutral', 'positive']
# )

tweet_eval_prompt = dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        f'Answer the question by replying A, B or C.\nTweet: "{{text}}"\n Choices: \nA. negative \nB. neutral \nC. positive\nAnswer:',
                    ),
                ],
            ),
            ice_token='</E>',
        )
tweet_eval_prompt_ice = dict(
    type=PromptTemplate,
    template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f'What is the sentiment of the following tweet? Answer the question by replying A, B or C.\nTweet: "{{text}}"\n Choices: \nA. negative \nB. neutral \nC. positive\nAnswer:',
                ),
                dict(role='BOT', prompt='{label}\n')
            ]),
)

tweeteval_sentiment = dict(
    type='TweetEvalSentimentDataset',
    abbr='tweeteval-sentiment',
    # path='data/tweeteval_sentiment.jsonl',  # not needed since we load from HF
    path='tweeteval',
    reader_cfg=dict(
        input_columns=['text'],
        output_column='label'
    ),
    infer_cfg=dict(
        ice_template=tweet_eval_prompt_ice,
        prompt_template=tweet_eval_prompt,
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer)
    ),
    eval_cfg=tweeteval_sentiment_evaluator
)

tweeteval_datasets = [tweeteval_sentiment]
