import os.path as path
from collections import defaultdict
from typing import Any, List, Optional

from datasets import load_dataset

from detection.arguments import form_args, get_dataset_path
from detection.utils import BinaryDataset, save_binary_dataset

SUPPORTED_DATASETS = ['mock', 'tatoeba', 'wikimatrix']

# --- Datasets configs description ---

CONFIGS = {
    'tatoeba': {
        'path': 'tatoeba',
    },
    'wikimatrix': {}
}
CONFIGS = defaultdict(dict, CONFIGS)


# --- Datasets sources description ---

def _empty_function(*args, **kwargs) -> None:
    pass


ENTRYPOINTS = {
    'tatoeba': load_dataset,
    'wikimatrix': _empty_function
}

# --- Using languages description ---
# We suppose that languages follow the order: [source language, target language]
# Every dataset has a default language pair (it should be put first).
# Then, generate.py should not support iterating through several languages.
# This logic is put directly into scripts.

DEFAULT_LANGS = ['de', 'en']
LANGS = {
    'tatoeba': [
        DEFAULT_LANGS,
        ['en', 'nl'],
        ['en', 'es'],
        ['en', 'he'],
        ['en', 'fi'],
        ['fr', 'ru'],
    ],
    'wikimatrix': [
        DEFAULT_LANGS,
    ]
}
LANGS = defaultdict(list, LANGS)


class DatasetFactory:
    @staticmethod
    def crop(dataset: BinaryDataset, dataset_name: str, size: Optional[int]) -> BinaryDataset:
        if not size:
            return dataset
        if dataset_name == 'tatoeba':
            # TODO-Extra: Figure out how to crop better
            # dataset['train']['translation'] = dataset['train']['translation'][:size]
            pass
        elif dataset_name == 'wikimatrix':
            pass
        return dataset

    @staticmethod
    def get(dataset_name: str, langs: List[str]) -> Any:
        config = CONFIGS[dataset_name]
        if dataset_name == 'tatoeba':
            config['lang1'], config['lang2'] = langs
        elif dataset_name == 'wikimatrix':
            pass
        entrypoint = ENTRYPOINTS[dataset_name]
        source_dataset = entrypoint(**config)
        source_dataset.dataset_name = dataset_name
        return source_dataset

    @staticmethod
    def get_languages(dataset_name: str) -> List[List[str]]:
        return LANGS[dataset_name]


def collect(chosen_dataset_name: str,
            save: bool = False,
            size: Optional[int] = None,
            ext: str = 'bin') -> List[BinaryDataset]:
    """
    Parameters
    ----------
        chosen_dataset_name: str
            One of the following datasets: ['mock', 'tatoeba', 'wikimatrix'].
        save: bool
            Flag showing should we save datasets or not.
        size: Optional[int]
            Common size of binary datasets.
        ext: str
            An extension for datasets dumped files names.

    Returns
    -------
        collection: List[BinaryDataset]
            List of datasets which are loaded before translations.
    """
    if chosen_dataset_name not in SUPPORTED_DATASETS:
        raise ValueError('Wrong chosen dataset name')

    collection = []
    langs = DatasetFactory.get_languages(chosen_dataset_name)
    for langs_pair in langs:
        if not langs_pair:
            continue
        print(f"Handling languages... Lang #1 = '{langs_pair[0]}'; Lang # 2 = '{langs_pair[1]}'...")
        source_dataset = DatasetFactory.get(chosen_dataset_name, langs_pair)
        source_dataset = DatasetFactory.crop(source_dataset, chosen_dataset_name, size)
        if save:
            save_binary_dataset(source_dataset, langs=langs_pair, ext=ext)
        if source_dataset:
            collection.append(source_dataset)
    return collection


if __name__ == '__main__':
    main_args = form_args()
    source_datasets = collect(main_args.dataset_name, save=True, ext=main_args.bin_ext)
    print('Source datasets length =', len(source_datasets))
