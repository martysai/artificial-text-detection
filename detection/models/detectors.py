# TODO-Pipeline: put Trainer and Tokenizer for DistillBERT here
import argparse
import os
from typing import Any, Dict, Optional

import pandas as pd
import torch
import transformers
from transformers import (
    DistilBertForSequenceClassification, DistilBertTokenizerFast, IntervalStrategy, Trainer, TrainingArguments,
    BertTokenizerFast, AutoModelForSequenceClassification
)
from detection.models.const import CLASSIFICATION_THRESHOLD, HF_MODEL_NAME
from detection.data.datasets import TextDetectionDataset
from detection.models.validate import compute_metrics


class Detector:
    @staticmethod
    def get_training_arguments(args: argparse.Namespace) -> TrainingArguments:
        path_to_resources = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        return TrainingArguments(
            evaluation_strategy=IntervalStrategy.EPOCH,
            output_dir=f"{path_to_resources}/resources/data/training_results",
            num_train_epochs=args.epochs,
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
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        device: Optional[str] = "cpu"
    ) -> TextDetectionDataset:
        data = X.copy()
        data["target"] = y if y is not None else ["human"] * len(data)
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
        use_wandb: Optional[bool] = False
    ):
        if (not model or not training_args) and (not args):
            raise AttributeError("Wrong parameters passed to SimpleDetector. Fill args")
        self.use_wandb = use_wandb
        args.report_to = ["wandb"] if self.use_wandb else []
        self.model = model or self.load_model(args)
        self.training_args = training_args or self.get_training_arguments(args)
        self.trainer = None
        self.tokenizer = BertTokenizerFast.from_pretrained(HF_MODEL_NAME)

    def load_model(self, args: Optional[argparse.Namespace]) -> Any:
        if hasattr(args, "model_path") and os.path.exists(args.model_path):
            model = transformers.PreTrainedModel.from_pretrained(args.model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                HF_MODEL_NAME,
                num_labels=1
            )
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
        self.trainer = self.load_trainer(train_dataset, eval_dataset)
        self.trainer.train()
        self.trainer.save_model()

    def get_logit(self, sample: Dict[str, Any]) -> float:
        sample.pop("labels", None)
        sample["input_ids"] = sample["input_ids"].view(1, -1)
        sample["attention_mask"] = sample["attention_mask"].view(1, -1)
        logit = self.trainer.model(**sample).logits[0][0].detach().cpu().numpy().reshape(-1)[0]
        return logit

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        dataset = self.convert_dataframe_to_dataset(X)
        preds = []
        with torch.no_grad():
            for sample in dataset:
                logit = self.get_logit(sample)
                preds.append("machine" if logit > CLASSIFICATION_THRESHOLD else "human")
        return pd.DataFrame(preds, columns=["target"])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        dataset = self.convert_dataframe_to_dataset(X)
        with torch.no_grad():
            probas = [self.get_logit(sample) for sample in dataset]
        return pd.DataFrame(probas, columns=["proba"])
