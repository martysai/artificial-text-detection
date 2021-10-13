import argparse
import os.path

import wandb

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from detection.models.validate import compute_metrics
from detection.data.generate import generate


def set_args(parser: argparse.ArgumentParser):
    runtime = parser.add_argument_group('Checkpoints')
    runtime.add_argument('--model_path', type=str, default='model.pth',
                         help='Model checkpoint path')
    return parser


def run(
    args,
    run_name: str = 'default',
) -> Trainer:
    train_dataset, eval_dataset = generate()

    training_args = TrainingArguments(
        evaluation_strategy='epoch',
        output_dir='./results',
        num_train_epochs=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        report_to='wandb',
        run_name=run_name,
    )

    if not os.path.exists(args.model_path):
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    else:
        # TODO: specify how to load a model
        pass

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    wandb.finish()

    trainer.save_model()

    return trainer


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        'Deep Learning Hometask 1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_args(arg_parser)
    known_args, _ = arg_parser.parse_known_args()
    run(known_args)
