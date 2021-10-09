import wandb

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from models.validate import compute_metrics
from data.generate import generate


def run(
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

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    wandb.finish()

    return trainer
