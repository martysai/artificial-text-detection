import os
import os.path as path
from typing import List

import transformers
import wandb
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from detection.models.validate import compute_metrics
from detection.arguments import form_args, get_dataset_path
from detection.data.factory import BinaryDataset, collect, save_binary_dataset
from detection.data.generate import generate
from detection.utils import TrainEvalDatasets
from detection.data.wrapper import TextDetectionDataset


def setup_experiment_tracking(args) -> None:
    token = os.environ.get('WANDB_TOKEN', None)
    wandb.login(key=token)
    wandb.init(project='artificial-text-detection', name=args.run_name)


def stop_experiment_tracking() -> None:
    wandb.finish()


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
        save_binary_dataset(binary_dataset, ext=args.bin_ext)
    return source_datasets


def load_translated_datasets(args) -> TrainEvalDatasets:
    train_dataset = TextDetectionDataset.load(args.dataset_path, suffix='train')
    train_dataset.device = args.device
    eval_dataset = TextDetectionDataset.load(args.dataset_path, suffix='eval')
    eval_dataset.device = args.device
    return train_dataset, eval_dataset


def translate_binary_datasets(datasets: List[BinaryDataset],
                              datasets_names: List[str],
                              args) -> TrainEvalDatasets:
    """
    A pipeline wrapper for generate method.
    TODO

    Parameters
    ----------
        datasets:
        datasets_names: List[str]
        args

    Returns
    -------
        train_dataset, eval_dataset: TrainEvalDatasets

    """
    # TODO: дописать функционал с разными языками
    dataset, dataset_name = datasets[0], datasets_names[0]
    train_path = get_dataset_path(args.dataset_name, f'train.{args.ds_ext}')
    eval_path = get_dataset_path(args.dataset_name, f'eval.{args.ds_ext}')
    if not (path.exists(train_path) and path.exists(eval_path)):
        train_dataset, eval_dataset = generate(dataset,
                                               dataset_name,
                                               device=args.device,
                                               size=args.size,
                                               ext=args.ds_ext)
    else:
        print('Datasets have already been processed. Paths: '
              f'dataset path = {args.dataset_path}')
        train_dataset, eval_dataset = load_translated_datasets(args)
    return train_dataset, eval_dataset


def train_text_detection_model(
        train_dataset: TextDetectionDataset,
        eval_dataset: TextDetectionDataset,
        args) -> Trainer:
    training_args = TrainingArguments(
        evaluation_strategy='epoch',
        output_dir='./results',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir='./logs',
        logging_steps=args.log_steps,
        report_to='wandb',
        run_name=args.run_name,
    )

    if not os.path.exists(args.model_path):
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
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


def pipeline(args) -> Trainer:
    """
    TODO
    """
    setup_experiment_tracking(args)

    datasets = create_binary_datasets(args)
    # TODO: extend for multiple dataset names
    datasets_names = [args.dataset_name]

    train_dataset, eval_dataset = translate_binary_datasets(datasets, datasets_names, args)

    trainer = train_text_detection_model(train_dataset, eval_dataset, args)
    stop_experiment_tracking()
    return trainer


if __name__ == '__main__':
    global_args = form_args()
    pipeline(global_args)
