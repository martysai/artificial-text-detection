from typing import Dict

import numpy as np
from datasets import load_metric
from transformers import EvalPrediction

from detection.models.const import CLASSIFICATION_THRESHOLD, METRIC_NAMES

METRICS = {metric_name: load_metric(metric_name) for metric_name in METRIC_NAMES}


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.array([int(logit > CLASSIFICATION_THRESHOLD) for logit in logits])
    metrics_args = {
        "predictions": predictions,
        "references": labels,
    }

    metrics_dict = {}
    for metric_name in METRIC_NAMES:
        metrics_dict.update(METRICS[metric_name].compute(**metrics_args))

    return metrics_dict
