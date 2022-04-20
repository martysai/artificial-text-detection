from datasets import load_metric

bleurt = load_metric("bleurt", "BLEURT-20")

# TODO: write down the bleurt value
print(bleurt)
# $HOME/.cache/huggingface/metrics/bleurt/BLEURT-20/downloads
