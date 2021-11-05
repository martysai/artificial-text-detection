import os
from typing import Tuple

import transformers
import wandb
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from detection.models.validate import compute_metrics
from detection.arguments import form_args
from detection.data.factory import collect
from detection.data.generate import generate
from detection.data.wrapper import TextDetectionDataset


def setup_experiment_tracking(args) -> None:
    token = os.environ.get('POETRY_WANDB_TOKEN', None)
    wandb.login(key=token)
    wandb.init(project='artificial-text-detection', name=args.run_name)


def stop_experiment_tracking() -> None:
    wandb.finish()


def create_binary_datasets(args) -> None:
    collect(args.dataset_name, save=True, ext=args.ext)


def load_translated_datasets(args) -> Tuple[TextDetectionDataset, TextDetectionDataset]:
    train_dataset = TextDetectionDataset.load(args.dataset_path, suffix='train')
    train_dataset.device = args.device
    eval_dataset = TextDetectionDataset.load(args.dataset_path, suffix='eval')
    eval_dataset.device = args.device
    return train_dataset, eval_dataset


def translate_binary_datasets(args) -> Tuple[TextDetectionDataset, TextDetectionDataset]:
    # TODO: можно ли их разделить на превращение и перевод?
    if not (os.path.exists(f'{args.dataset_path}.train') and os.path.exists(f'{args.dataset_path}.eval')):
        train_dataset, eval_dataset = generate(size=args.size, dataset_path=args.dataset_path,
                                               is_mock_data=args.is_mock_data)
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
    setup_experiment_tracking(args)

    create_binary_datasets(args)

    train_dataset, eval_dataset = translate_binary_datasets(args)

    trainer = train_text_detection_model(train_dataset, eval_dataset, args)
    stop_experiment_tracking()
    return trainer


if __name__ == '__main__':
    main_args = form_args()
    pipeline(main_args)
