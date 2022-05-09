import argparse
import os
import os.path as path
from typing import List

import transformers
from transformers import AutoModelForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments

from artificial_detection.arguments import form_args
from artificial_detection.data.data import BinaryDataset, TextDetectionDataset
from artificial_detection.data.factory import DatasetFactory, collect
from artificial_detection.data.generate import generate
from artificial_detection.models.const import HF_MODEL_NAME
from artificial_detection.models.validate import compute_metrics
from artificial_detection.utils import (
    get_dataset_path,
    save_binary_dataset,
    setup_experiment_tracking,
    stop_experiment_tracking,
    translations_to_torch_dataset,
)


def create_binary_datasets(args) -> List[BinaryDataset]:
    """
    A pipeline wrapper for collect method.

    Parameters
    ----------
        args
            Set of arguments

    Returns
    -------
        Collected binary datasets.
    """
    source_datasets = collect(args.dataset_name, save=True, size=args.size, ext=args.bin_ext)
    for binary_dataset in source_datasets:
        save_binary_dataset(binary_dataset, args.dataset_name, ext=args.bin_ext)
    return source_datasets


def translate_binary_datasets(
    datasets: List[BinaryDataset], datasets_names: List[str], args
) -> List[TextDetectionDataset]:
    """
    A pipeline wrapper for generate method.
    TODO-Doc

    Parameters
    ----------
        datasets:
        datasets_names: List[str]
        args

    Returns
    -------
        dataset: TextDetectionDataset

    """
    # TODO-WikiMatrix: improve for multiple datasets
    dataset_name = datasets_names[0]
    languages = DatasetFactory.get_languages(dataset_name)

    translated_datasets = []
    for i, lang_pair in enumerate(languages):
        dataset = datasets[i]
        src_lang, trg_lang, direction = lang_pair
        if direction == "reversed":
            src_lang, trg_lang = trg_lang, src_lang
        elif direction != "straight":
            raise ValueError("Wrong direction passed to language pairs")
        csv_path = get_dataset_path(f"{dataset_name}.{src_lang}-{trg_lang}", ext="csv")
        if not path.exists(csv_path):
            generated_dataset = generate(
                dataset,
                dataset_name,
                src_lang=src_lang,
                trg_lang=trg_lang,
                size=args.size,
                model_name=args.model_name,
                device=args.device,
                batch_size=args.easy_nmt_batch_size,
                easy_nmt_offline=args.easy_nmt_offline,
                offline_prefix=args.offline_prefix,
                offline_cache_prefix=args.offline_cache_prefix
            )
            text_detection_dataset = translations_to_torch_dataset(
                generated_dataset.targets,
                generated_dataset.translations,
                device=args.device,
                easy_nmt_offline=args.easy_nmt_offline
            )
        else:
            print(f"This dataset has already been processed. CSV Path = {csv_path}")
            tokenizer = BertTokenizerFast.from_pretrained(HF_MODEL_NAME)
            text_detection_dataset = TextDetectionDataset.load_csv(csv_path, tokenizer, device=args.device)
        translated_datasets.append(text_detection_dataset)
    return translated_datasets


def train_text_detection_model(dataset: TextDetectionDataset, args) -> Trainer:
    train_dataset, eval_dataset = dataset.split()

    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_steps=args.log_steps,
        report_to="wandb",
        run_name=args.run_name,
    )

    if not os.path.exists(args.model_path):
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME, num_labels=1)
    else:
        model = transformers.PreTrainedModel.from_pretrained(args.model_path)
    model = model.to(args.device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()
    return trainer


def pipeline(args: argparse.Namespace) -> List[Trainer]:
    """
    TODO-Doc
    """
    datasets = create_binary_datasets(args)
    # TODO-WikiMatrix: extend for multiple dataset names
    datasets_names = [args.dataset_name] * len(datasets)

    dataset_with_langs = translate_binary_datasets(datasets, datasets_names, args)

    trainers = []
    for dataset in dataset_with_langs:
        setup_experiment_tracking(args.run_name)
        trainer = train_text_detection_model(dataset, args)
        trainers.append(trainer)
        stop_experiment_tracking()
    return trainers


if __name__ == "__main__":
    global_args = form_args()
    pipeline(global_args)
