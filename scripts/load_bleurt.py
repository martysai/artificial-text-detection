from datasets import load_metric

bleurt = load_metric("bleurt", "BLEURT-20")

print(bleurt)
# $HOME/.cache/huggingface/metrics/bleurt/BLEURT-20
