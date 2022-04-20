from artificial_detection.data.proxy import CometMetrics

comet_model, model_path = CometMetrics.load_offline()

print("model_path:", model_path)
# $HOME/.cache/torch/unbabel_comet/wmt20-comet-da/checkpoints/model.ckpt
