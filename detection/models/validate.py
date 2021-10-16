import numpy as np

from datasets import load_metric
from typing import Dict
from transformers import EvalPrediction


METRIC_NAMES = [
    'accuracy',
    'f1',
    'precision',
    'recall'
]
METRICS = {
    metric_name: load_metric(metric_name)
    for metric_name in METRIC_NAMES
}


def compute_metrics(
    eval_pred: EvalPrediction
) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics_args = {
        'predictions': predictions,
        'references': labels,
    }

    # metrics_dict = {
    #     metric_name: METRICS[metric_name].compute(**metrics_args)
    #     for metric_name in METRIC_NAMES
    # }
    # metrics_dict = {
    #     metric_name: next(iter(value_dict.values()))
    #     for metric_name, value_dict in metrics_dict.items()
    # }
    metrics_dict = {}
    for metric_name in METRIC_NAMES:
        metrics_dict.update(METRICS[metric_name].compute(**metrics_args))

    return metrics_dict
