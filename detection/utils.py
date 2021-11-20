from typing import List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

from detection.arguments import get_dataset_path
from detection.data.wrapper import TextDetectionDataset


SRC_LANG = 'de'
TRG_LANG = 'en'

SAVING_FREQ = 50000
LOGGING_FREQ = 250

TEST_SIZE = 0.2

TrainEvalDatasets = Tuple[TextDetectionDataset, TextDetectionDataset]


class MockDataset:
    dataset = [
        {
            SRC_LANG: 'guten tag',
            TRG_LANG: 'good evening',
        },
        {
            SRC_LANG: 'es tut mir leid',
            TRG_LANG: 'i am sorry',
        }
    ]
    _translations = ['good evening', 'i am sorry']
    dataset_name = 'mock'

    @classmethod
    def targets(cls) -> List[str]:
        return [sample[TRG_LANG] for sample in cls.dataset]

    @classmethod
    def translations(cls) -> List[str]:
        return cls._translations

    @classmethod
    def list(cls) -> List[str]:
        dataset_list = []
        for dct in cls.dataset:
            dataset_list.extend([dct[SRC_LANG], dct[TRG_LANG]])
        return dataset_list


def log(index: int, length: int, src_sample: str, tld_sample: str) -> None:
    if (index + 1) % LOGGING_FREQ == 0:
        print(f'[{index + 1}/{length}] Preprocessing sample = {src_sample}\n'
              f'[{index + 1}/{length}] Translated sample = {tld_sample}\n')


def translations_list_to_dataset(
        targets: List[str],
        translations: List[str],
        device: Optional[str] = None) -> TrainEvalDatasets:
    corpus = [''] * (2 * len(targets))
    corpus[::2] = translations
    corpus[1::2] = targets
    labels = torch.FloatTensor([0, 1] * len(targets))
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        corpus, labels, test_size=TEST_SIZE
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


def save_translations_texts(
        sources: List[str],
        targets: List[str],
        translations: List[str],
        dataset_name: str) -> None:
    print('Saving sources/translations in csv...')
    df_data = list(zip(sources, targets, translations))
    df = pd.DataFrame(data=df_data, columns=['sources', 'targets', 'translations'])
    csv_path = get_dataset_path(dataset_name, ext='csv')
    df.to_csv(csv_path, index=False)


def save_translations(
        targets: List[str],
        translations: List[str],
        dataset_name: str,
        device: str,
        ext: str) -> TrainEvalDatasets:
    print('Saving translations in pth...')
    train_dataset, eval_dataset = translations_list_to_dataset(targets, translations, device=device)
    train_path = get_dataset_path(f'{dataset_name}.train', ext=ext)
    eval_path = get_dataset_path(f'{dataset_name}.eval', ext=ext)
    train_dataset.save(train_path)
    eval_dataset.save(eval_path)
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
