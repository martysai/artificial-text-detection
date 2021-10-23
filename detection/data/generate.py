import logging
import pickle
from typing import Callable, List, Tuple

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
LOGGING_FREQ = 250


def get_buffer(
        dataset,
        transform: Callable
) -> List[str]:
    buffer = []
    for i, sample in enumerate(dataset):
        src, trg = sample[SRC_LANG], sample[TRG_LANG]
        if (i + 1) % LOGGING_FREQ == 0:
            logging.info(f'[{i + 1}/{len(dataset)}] Preprocessing sample = {src}')
            print(f'[{i + 1}/{len(dataset)}] Preprocessing sample = {src}')
        gen = transform(src)
        buffer.extend([gen, trg])
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

    buffer = get_buffer(dataset, model)
    labels = [0, 1] * len(dataset)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        buffer, labels, test_size=TEST_SIZE
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = TextDetectionDataset(train_encodings, train_labels)
    eval_dataset = TextDetectionDataset(eval_encodings, val_labels)

    train_dataset.save(dataset_path, suffix='train')
    eval_dataset.save(dataset_path, suffix='eval')

    return train_dataset, eval_dataset


if __name__ == '__main__':
    main_args = form_args()
    generate(main_args.size)
