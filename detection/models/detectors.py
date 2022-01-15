# TODO-Pipeline: put Trainer and Tokenizer for DistillBERT here
import argparse
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import transformers
from scipy import special
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

from detection.data.datasets import TextDetectionDataset
from detection.models.const import CLASSIFICATION_THRESHOLD, HF_MODEL_NAME, HF_MODEL_PATH
from detection.models.validate import compute_metrics
from detection.utils import setup_experiment_tracking, stop_experiment_tracking


class Detector:
    @staticmethod
    def get_training_arguments(args: argparse.Namespace) -> TrainingArguments:
        path_to_resources = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        return TrainingArguments(
            evaluation_strategy=IntervalStrategy.EPOCH,
            output_dir=f"{path_to_resources}/resources/data/training_results",
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch,
            per_device_eval_batch_size=args.eval_batch,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            logging_dir="./logs",
            logging_steps=args.log_steps,
            report_to=args.report_to,
            run_name=args.run_name,
        )

    def convert_dataframe_to_dataset(
        self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, device: Optional[str] = "cpu"
    ) -> TextDetectionDataset:
        data = X.copy()
        data["target"] = y if y is not None else ["human"] * len(data)  # TODO: remove this strange part
        return TextDetectionDataset.load_csv(data, tokenizer=self.tokenizer, device=device, new=True)

    def load_model(self, args: Optional[argparse.Namespace]) -> Any:
        raise NotImplementedError

    def load_trainer(self, train_dataset: TextDetectionDataset, eval_dataset: TextDetectionDataset) -> Trainer:
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class SimpleDetector(Detector):
    # TODO-Unsupervised: Check the article -- which detector to use
    def __init__(
        self,
        model: Optional[Any] = None,
        training_args: Optional[TrainingArguments] = None,
        args: Optional[argparse.Namespace] = None,
        use_wandb: Optional[bool] = False,
    ):
        if (not model or not training_args) and (not args):
            raise AttributeError("Wrong parameters passed to SimpleDetector. Fill args")
        self.offline = hasattr(args, "model_path") and os.path.exists(args.model_path)
        self.model_path = HF_MODEL_PATH if self.offline else HF_MODEL_NAME
        self.run_name = args.run_name
        self.use_wandb = use_wandb
        args.report_to = ["wandb"] if self.use_wandb else []
        self.model = model or self.load_model(args)
        self.training_args = training_args or self.get_training_arguments(args)
        self.trainer = None
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)

    def load_model(self, args: Optional[argparse.Namespace]) -> Any:
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=1)
        if hasattr(args, "device"):
            model = model.to(args.device)
        return model

    def load_trainer(self, train_dataset: TextDetectionDataset, eval_dataset: TextDetectionDataset) -> Trainer:
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        dataset = self.convert_dataframe_to_dataset(X, y)
        train_dataset, eval_dataset = dataset.split()
        setup_experiment_tracking(self.run_name)
        self.trainer = self.load_trainer(train_dataset, eval_dataset)
        self.trainer.train()
        stop_experiment_tracking()
        self.trainer.save_model()

    def get_logit(self, sample: Dict[str, Any]) -> float:
        sample.pop("labels", None)
        sample["input_ids"] = sample["input_ids"].view(1, -1)
        sample["attention_mask"] = sample["attention_mask"].view(1, -1)
        logit = self.trainer.model(**sample).logits[0][0].detach().cpu().numpy().reshape(-1)[0]
        return logit

    @staticmethod
    def get_probits(logits: np.ndarray) -> np.ndarray:
        return special.softmax(logits, axis=0)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        dataset = self.convert_dataframe_to_dataset(X)
        with torch.no_grad():
            logits = np.array([self.get_logit(sample) for sample in dataset]).reshape(-1, 1)
            preds = ["machine" if logit > CLASSIFICATION_THRESHOLD else "human" for logit in logits]
        return pd.DataFrame(preds, columns=["target"])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        dataset = self.convert_dataframe_to_dataset(X)
        with torch.no_grad():
            logits = np.array([self.get_logit(sample) for sample in dataset]).reshape(-1, 1)
        return pd.DataFrame(logits, columns=["proba"])
