from datasets import load_metric

bert_score_metric = load_metric("bertscore")
print(bert_score_metric)

# $HOME/.cache/huggingface/metrics/bert_score
