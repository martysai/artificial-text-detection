import argparse
import os

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
    train_args.add_argument('--size', type=int, default=10000,
                            help='# samples in the tatoeba dataset')
    train_args.add_argument('--warmup_steps', type=int, default=100)
    train_args.add_argument('--weight_decay', type=int, default=1e-4)

    libraries = parser.add_argument_group('Libraries')
    libraries.add_argument('--wandb_path', type=str, default='wandb.key',
                           help='A path to wandb personal token')
    libraries.add_argument('--run_name', type=str, default='default',
                           help='A name of run to be used in wandb')

    return parser


def run(args) -> Trainer:
    working_dir = os.path.dirname(os.getcwd())
    wandb_path = os.path.join(working_dir, args.wandb_path)
    if not os.path.exists(wandb_path):
        raise FileNotFoundError('Put wandb personal token into '
                                f"args.wandb_path = '{wandb_path}'")
    with open(wandb_path, 'r') as wandb_file:
        token = wandb_file.read()
        wandb.login(key=token)
        wandb.init(project='text-detection')
        wandb.run.name = args.run_name
        wandb.run.save()

    train_dataset, eval_dataset = generate(size=args.size)

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
