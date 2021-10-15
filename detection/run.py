import argparse
import os.path

import torch
import wandb

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from detection.models.validate import compute_metrics
from detection.data.generate import generate


def set_args(parser: argparse.ArgumentParser):
    checkpoints = parser.add_argument_group('Checkpoints')
    checkpoints.add_argument('--model_path', type=str, default='model.pth',
                             help='Model checkpoint path')
    train_args = parser.add_argument_group('Training arguments')
    train_args.add_argument('--epochs', type=int, default=50,
                            help='# epochs')
    train_args.add_argument('--train_batch', type=int, default=512,
                            help='train batch size')
    train_args.add_argument('--eval_batch', type=int, default=512,
                            help='eval batch size')
    train_args.add_argument('--log_steps', type=int, default=10,
                            help='# steps for logging')
    train_args.add_argument('--warmup_steps', type=int, default=100)
    train_args.add_argument('--weight_decay', type=int, default=1e-4)

    return parser


def run(
    args,
    run_name: str = 'default',
) -> Trainer:
    train_dataset, eval_dataset = generate()

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
        'Text Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_args(arg_parser)
    known_args, _ = arg_parser.parse_known_args()
    known_args.cuda = torch.cuda.is_available()
    known_args.device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.
                                     cuda.is_available() else 'cpu')
    run(known_args)
