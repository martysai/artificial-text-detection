import argparse
from functools import partial
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainerCallback, XLMRobertaForSequenceClassification

from artificial_detection.arguments import form_supervised_args
from artificial_detection.pipelines.compute import compute_metrics
from artificial_detection.utils import setup_experiment_tracking, stop_experiment_tracking


def preprocess_examples(examples, tokenizer):
    result = tokenizer(
        examples['text'],
        truncation=True, padding="max_length", max_length=64
    )
    result["label"] = examples["label"]
    return result


def read_splits(df, as_datasets):
    train_df = df[df["subset"] == "tr"]
    dev_df = df[df["subset"] == "va"]
    test_df = df[df["subset"] == "public"]

    if as_datasets:
        train, dev, test = map(Dataset.from_pandas, (train_df, dev_df, test_df))
        return DatasetDict(train=train, dev=dev, test=test)
    else:
        return train_df, dev_df, test_df


def prepare_data(tokenizer):
    data_path = str(Path(__file__).parents[3] / "atd-data/metrics_df.tsv")
    df = pd.read_csv(data_path, sep="\t")

    splits = read_splits(df, True)
    splits = splits.shuffle(seed=42)
    tokenized_splits = splits.map(
        partial(preprocess_examples, tokenizer=tokenizer),
        batched=True,
    )
    return tokenized_splits


class StopCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.global_step <= 700:
            return
        if metrics["eval_recall"] == 1.0 or metrics["eval_auc"] <= 0.51:
            raise ValueError("bad metrics")
        if metrics["eval_accuracy"] <= 0.68 and state.global_step >= 1200:
            raise ValueError("weak hyper parameters")
        if metrics["eval_accuracy"] <= 0.72 and state.global_step >= 2700:
            raise ValueError("weak hyper parameters")


prefix = Path(__file__).parents[3]
model_name_to_path = {
    "XLM": str(prefix / "atd-models/xlm-roberta-large"),
}


def load_detector(model_name: str, max_length: int = 64):
    if model_name not in model_name_to_path:
        raise ValueError(f"Model {model_name} not found")
    model_path = model_name_to_path[model_name]
    if model_name == "XLM":
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length)
        model = XLMRobertaForSequenceClassification.from_pretrained(
            model_path, max_length=max_length
        )
    else:
        raise ValueError(f"Model {model_name} not found")
    return tokenizer, model


def set_trainer(tokenized_splits, tokenizer, model, warmup_ratio, lr_scheduler_type, optim, lr, weight_decay):
    es_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
    )
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        optim=optim,
        logging_dir='./logs',
        report_to="wandb",
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=300,
        eval_steps=300,
        save_steps=900,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        save_total_limit=10,
        run_name=f"Ru RoBERTa LR={lr} OPTIM={optim} LR_SCH_TYPE={lr_scheduler_type}, WR={warmup_ratio}"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_splits['train'],
        eval_dataset=tokenized_splits['dev'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[StopCallback(), es_callback]
    )
    return trainer


class SupervisedPipeline:
    def __init__(self, run_name: str):
        self.run_name = run_name

    def train(
        self,
        warmup_ratio: float = 0.01,
        lr_scheduler_type: str = "cosine_with_restarts",
        optim: str = "adafactor",
        lr: float = 1e-5,
        weight_decay: float = 0.01,
    ):
        setup_experiment_tracking(run_name=self.run_name)

        tokenizer, model = load_detector(model_name=args.model_name)
        tokenized_splits = prepare_data(tokenizer)

        trainer = set_trainer(
            tokenized_splits=tokenized_splits,
            tokenizer=tokenizer,
            model=model,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            optim=optim,
            lr=lr,
            weight_decay=weight_decay
        )
        trainer.train()

        stop_experiment_tracking()

        return model


def grid_search():
    pass


def main(args: argparse.Namespace) -> None:
    pipeline = SupervisedPipeline(run_name=args.run_name)
    # pipeline.init(args)
    trainer = pipeline.train()
    # collected_results = pipeline.evaluate(trainer)
    # pipeline.save(trainer, collected_results)


if __name__ == "__main__":
    supervised_args = form_supervised_args()
    main(supervised_args)
