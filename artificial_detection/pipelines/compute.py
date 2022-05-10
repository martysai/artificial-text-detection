import datasets
import numpy as np
from datasets import load_metric
from sklearn.metrics import auc, brier_score_loss, fbeta_score, precision_recall_curve
from transformers import EvalPrediction


class FBeta(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=[],
        )

    def _compute(
        self, predictions, references, beta=1.0, labels=None, pos_label=1, average="binary", sample_weight=None
    ):
        score = fbeta_score(
            references, predictions, beta=beta, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight
        )
        return {"fbeta": float(score) if score.size == 1 else score}


class PRAUC(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "prediction_scores": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Value("int32"),
                }
                if self.config_name == "multiclass"
                else {
                    "references": datasets.Sequence(datasets.Value("int32")),
                    "prediction_scores": datasets.Sequence(datasets.Value("float")),
                }
                if self.config_name == "multilabel"
                else {
                    "references": datasets.Value("int32"),
                    "prediction_scores": datasets.Value("float"),
                }
            ),
            reference_urls=[],
        )

    def _compute(
        self,
        references,
        prediction_scores,
        pos_label=1,
        sample_weight=None,
    ):
        precision, recall, thresholds = precision_recall_curve(references, prediction_scores, pos_label=pos_label, sample_weight=sample_weight)
        auc_precision_recall = auc(recall, precision)
        return {"pr_auc": auc_precision_recall}


class BRIER(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(self._get_feature_types()),
            reference_urls=[],
        )

    def _get_feature_types(self):
        if self.config_name == "multilist":
            return {
                "predictions": datasets.Sequence(datasets.Value("float")),
                "references": datasets.Sequence(datasets.Value("float")),
            }
        else:
            return {
                "predictions": datasets.Value("float"),
                "references": datasets.Value("float"),
            }

    def _compute(self, predictions, references, sample_weight=None, pos_label=1):
        brier = brier_score_loss(
            references, predictions, sample_weight=sample_weight, pos_label=pos_label
        )

        return {"brier": brier}


ACCURACY = load_metric("accuracy")
AUC = load_metric("roc_auc")
F1 = load_metric("f1")
FBETA = FBeta()
PR_AUC = PRAUC()
BRIER_SCORE = BRIER()
PRECISION = load_metric("precision")
RECALL = load_metric("recall")


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)

    acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)
    precision_result = PRECISION.compute(predictions=preds, references=p.label_ids)
    recall_result = RECALL.compute(predictions=preds, references=p.label_ids)

    auc_result = AUC.compute(prediction_scores=preds, references=p.label_ids)

    f1_result = F1.compute(predictions=preds, references=p.label_ids)

    f05_result = FBETA.compute(predictions=preds, references=p.label_ids, beta=0.5)
    f2_result = FBETA.compute(predictions=preds, references=p.label_ids, beta=2.0)

    pr_auc = PR_AUC.compute(prediction_scores=preds, references=p.label_ids)

    brier = BRIER_SCORE.compute(predictions=preds, references=p.label_ids)

    result = {
        'precision': precision_result["precision"],
        'recall': recall_result["recall"],
        'accuracy': acc_result["accuracy"],
        'pr_auc': pr_auc["pr_auc"],
        'auc': auc_result["roc_auc"],
        'fhalf': f05_result["fbeta"],
        'f1': f1_result["f1"],
        'f2': f2_result["fbeta"],
        'brier': brier["brier"]
    }
    return result
