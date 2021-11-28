from collections import defaultdict
from typing import Any, Dict, List, Optional

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

def process_source_wikimatrix(path: str) -> List[str]:
    with open(path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def load_wikimatrix(lang1: str, lang2: str) -> List[Dict[str, str]]:
    sources_path = get_dataset_path(f'wikimatrix/WikiMatrix.{lang1}-{lang2}.txt', ext=lang1)
    sources = process_source_wikimatrix(sources_path)
    targets_path = get_dataset_path(f'wikimatrix/WikiMatrix.{lang1}-{lang2}.txt', ext=lang2)
    targets = process_source_wikimatrix(targets_path)
    dataset = [{lang1: sources[i], lang2: targets[i]} for i in list(range(len(sources)))]
    return dataset


ENTRYPOINTS = {
    'tatoeba': load_dataset,
    'wikimatrix': load_wikimatrix
}

# --- Using languages description ---
# We suppose that languages follow the order: [source language, target language]
# Every dataset has a default language pair (it should be put first).
# Then, generate.py should not support iterating through several languages.
# This logic is put directly into scripts.
# The third argument can take two possible values: ['straight', 'reversed'].

DEFAULT_LANGS = ['en', 'ru', 'straight']
LANGS = {
    'tatoeba': [
        DEFAULT_LANGS,
        ['es', 'ru', 'straight'],
        ['fi', 'ru', 'straight'],
        ['fr', 'ru', 'straight'],
    ],
    'wikimatrix': [
        DEFAULT_LANGS,
        ['es', 'ru', 'reversed'],
        ['fi', 'ru', 'reversed'],
        ['fr', 'ru', 'reversed'],
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
            dataset = dataset[:size]
        return dataset

    @staticmethod
    def get(dataset_name: str, langs: List[str]) -> Any:
        config = CONFIGS[dataset_name]
        if (dataset_name == 'tatoeba') or (dataset_name == 'wikimatrix'):
            config['lang1'], config['lang2'] = langs[0], langs[1]
        entrypoint = ENTRYPOINTS[dataset_name]
        source_dataset = entrypoint(**config)
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
            save_binary_dataset(source_dataset, chosen_dataset_name, langs=langs_pair, ext=ext)
        if source_dataset:
            collection.append(source_dataset)
    return collection


if __name__ == '__main__':
    main_args = form_args()
    source_datasets = collect(main_args.dataset_name, save=True, ext=main_args.bin_ext)
    print('Source datasets length =', len(source_datasets))
