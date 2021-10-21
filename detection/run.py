import logging
import os
import sys

import datasets
import transformers
import wandb

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from detection.models.validate import compute_metrics
from detection.data.generate import generate, extract_dataset
from detection.data.arguments import form_args


def setup_logging(training_args: TrainingArguments) -> None:
    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # set the main code and the modules it uses to the same log-level according to the node
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)


def run(args) -> Trainer:
    working_dir = os.path.dirname(os.getcwd())
    wandb_path = os.path.join(working_dir, args.wandb_path)
    if not os.path.exists(wandb_path):
        raise FileNotFoundError('Put wandb personal token into '
                                f"args.wandb_path = '{wandb_path}'")
    with open(wandb_path, 'r') as wandb_file:
        token = wandb_file.read()
        wandb.login(key=token)
        wandb.init(project='text-detection', name=args.run_name)

    if not (os.path.exists(f'{args.dataset_path}.train') or os.path.exists(f'{args.dataset_path}.eval')):
        train_dataset, eval_dataset = generate(size=args.size, dataset_path=args.dataset_path)
    else:
        logging.info('Datasets have already been processed. Paths: '
                     f'dataset path = {args.dataset_path}')
        train_dataset, eval_dataset = extract_dataset(args.dataset_path)

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
    setup_logging(training_args)

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
    main_args = form_args()
    run(main_args)
