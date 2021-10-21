import logging
import pickle
from typing import Callable, List, Tuple

from sklearn.model_selection import train_test_split
from datasets import load_dataset

from detection.models.translation import TranslationModel
from detection.data.dataset import TextDetectionDataset
from detection.data.arguments import form_args

from transformers import DistilBertTokenizerFast


DATASET = 'tatoeba'
SRC_LANG = 'ru'
TRG_LANG = 'en'
TEST_SIZE = 0.2
DATASET_SIZE = 10000
LOGGING_FREQ = 250


def extract_dataset(dataset_path: str) -> Tuple[TextDetectionDataset, TextDetectionDataset]:
    with open(f'{dataset_path}.train', 'rb') as train_file:
        train_dataset = pickle.load(train_file)
    with open(f'{dataset_path}.eval', 'rb') as eval_file:
        eval_dataset = pickle.load(eval_file)
    return train_dataset, eval_dataset


def save_dataset(dataset_path: str, train_dataset: TextDetectionDataset,
                 eval_dataset: TextDetectionDataset) -> None:
    if dataset_path:
        with open(f'{dataset_path}.train', 'wb') as train_file:
            pickle.dump(train_dataset, train_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{dataset_path}.eval', 'wb') as eval_file:
            pickle.dump(eval_dataset, eval_file, protocol=pickle.HIGHEST_PROTOCOL)


def get_buffer(
        dataset,
        transform: Callable,
        dataset_path: str = None
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
             dataset_path: str = None) -> Tuple[TextDetectionDataset, TextDetectionDataset]:
    dataset = load_dataset(DATASET, lang1=TRG_LANG, lang2=SRC_LANG)
    dataset = dataset['train'][:size]['translation']
    model = TranslationModel()

    buffer = get_buffer(dataset, model, dataset_path)
    labels = [0, 1] * len(dataset)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        buffer, labels, test_size=TEST_SIZE
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = TextDetectionDataset(train_encodings, train_labels)
    eval_dataset = TextDetectionDataset(eval_encodings, val_labels)

    save_dataset(dataset_path, train_dataset, eval_dataset)

    return train_dataset, eval_dataset


if __name__ == '__main__':
    main_args = form_args()
    generate(main_args.size)
