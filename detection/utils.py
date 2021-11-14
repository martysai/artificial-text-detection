from typing import List, Optional, Tuple

import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

from detection.arguments import get_dataset_path
from detection.data.wrapper import TextDetectionDataset


SRC_LANG = 'ru'
TRG_LANG = 'en'

SAVING_FREQ = 50000
LOGGING_FREQ = 250

TEST_SIZE = 0.2

TrainEvalDatasets = Tuple[TextDetectionDataset, TextDetectionDataset]


def get_mock_dataset() -> List[dict]:
    dataset = [
        {
            SRC_LANG: 'добрый вечер',
            TRG_LANG: 'good evening',
        },
        {
            SRC_LANG: 'прошу прощения',
            TRG_LANG: 'i am sorry',
        }
    ]
    return dataset


def get_mock_dataset_list() -> List[str]:
    dataset = get_mock_dataset()
    dataset_list = []
    for dct in dataset:
        dataset_list.extend([dct[SRC_LANG], dct[TRG_LANG]])
    return dataset_list


def log(index: int, length: int, sample: str) -> None:
    if (index + 1) % LOGGING_FREQ == 0:
        print(f'[{index + 1}/{length}] Preprocessing sample = {sample}')


def translations_list_to_dataset(
        translations: List[str],
        device: Optional[str] = None) -> Tuple[TextDetectionDataset, TextDetectionDataset]:
    labels = torch.FloatTensor([0, 1] * (len(translations) // 2))
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        translations, labels, test_size=TEST_SIZE
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(val_texts, truncation=True, padding=True)

    if device:
        # TODO: написать на GPU получше
        try:
            train_encodings = train_encodings.to(device)
            eval_encodings = eval_encodings.to(device)
            train_labels = train_labels.to(device)
            val_labels = val_labels.to(device)
        except AttributeError:
            pass

    train_dataset = TextDetectionDataset(train_encodings, train_labels)
    eval_dataset = TextDetectionDataset(eval_encodings, val_labels)

    return train_dataset, eval_dataset


def save_translations(
        translations: List[str],
        dataset_name: str,
        device: str,
        ext: str) -> TrainEvalDatasets:
    dataset_path = get_dataset_path(dataset_name, ext=ext)
    train_dataset, eval_dataset = translations_list_to_dataset(translations, device=device)
    train_dataset.save(dataset_path)
    eval_dataset.save(dataset_path)
    return train_dataset, eval_dataset


def save(translations: List[str],
         dataset_name: str,
         index: int,
         length: int,
         device: str,
         ext: str = 'bin') -> Optional[TrainEvalDatasets]:
    if (index + 1) % SAVING_FREQ == 0 and dataset_name:
        print(f'[{index + 1}/{length}] Saving a dataset...')
        return save_translations(translations, dataset_name, device, ext)
