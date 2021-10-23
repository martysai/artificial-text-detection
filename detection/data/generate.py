import logging
from typing import Callable, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from datasets import load_dataset

from detection.models.translation import TranslationModel
from detection.data.dataset import TextDetectionDataset
from detection.data.arguments import form_args
from detection.utils import get_mock_dataset

from transformers import DistilBertTokenizerFast


DATASET = 'tatoeba'
SRC_LANG = 'ru'
TRG_LANG = 'en'
TEST_SIZE = 0.2
DATASET_SIZE = 10000
SAVING_FREQ = 50000
LOGGING_FREQ = 250


def buffer2dataset(buffer: List[str]) -> Tuple[TextDetectionDataset, TextDetectionDataset]:
    labels = [0, 1] * (len(buffer) // 2)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        buffer, labels, test_size=TEST_SIZE
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = TextDetectionDataset(train_encodings, train_labels)
    eval_dataset = TextDetectionDataset(eval_encodings, val_labels)

    return train_dataset, eval_dataset


def get_buffer(
        dataset,
        transform: Callable,
        dataset_path: Optional[str] = None
) -> List[str]:
    buffer = []
    for i, sample in enumerate(dataset):
        src, trg = sample[SRC_LANG], sample[TRG_LANG]
        if (i + 1) % LOGGING_FREQ == 0:
            logging_message = f'[{i + 1}/{len(dataset)}] Preprocessing sample = {src}'
            logging.info(logging_message)
            print(logging_message)
        gen = transform(src)
        buffer.extend([gen, trg])
        if (i + 1) % SAVING_FREQ == 0 and dataset_path:
            saving_message = f'[{i + 1}/{len(dataset)}] Saving a dataset...'
            logging.info(saving_message)
            print(saving_message)
            train_dataset, eval_dataset = buffer2dataset(buffer)
            train_dataset.save(dataset_path, suffix='train')
            eval_dataset.save(dataset_path, suffix='eval')

    return buffer


def generate(size: int = DATASET_SIZE,
             dataset_path: str = None,
             is_mock_data: bool = False) -> Tuple[TextDetectionDataset, TextDetectionDataset]:
    if is_mock_data:
        dataset = get_mock_dataset()
    else:
        dataset = load_dataset(DATASET, lang1=TRG_LANG, lang2=SRC_LANG)
        dataset = dataset['train'][:size]['translation']
    model = TranslationModel()

    buffer = get_buffer(dataset, model, dataset_path=dataset_path)
    train_dataset, eval_dataset = buffer2dataset(buffer)
    train_dataset.save(dataset_path, suffix='train')
    eval_dataset.save(dataset_path, suffix='eval')

    return train_dataset, eval_dataset


if __name__ == '__main__':
    main_args = form_args()
    generate(main_args.size)
