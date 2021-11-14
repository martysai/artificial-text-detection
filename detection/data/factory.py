import pickle
import zlib
from collections import defaultdict
from typing import Any, List, Optional, Union

from datasets import dataset_dict, load_dataset

from detection.arguments import form_args, get_dataset_path

BinaryDataset = Union[Any, dataset_dict.DatasetDict]
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
DEFAULT_LANGS = ['en', 'ru']
LANGS = {
    'tatoeba': [
        DEFAULT_LANGS,
        [],
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
            # Figure out how to crop better
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
    def get_languages(dataset_name: str) -> Optional[List[str]]:
        # TODO: extend for multiple language pairs
        # for langs_pair in LANGS[dataset_name]:
        #     if langs_pair:
        #         return langs_pair
        return DEFAULT_LANGS


def load_binary_dataset(dataset_name: str, ext: str = 'bin') -> BinaryDataset:
    dataset_path = get_dataset_path(dataset_name, ext=ext)
    with open(dataset_path, 'rb') as file:
        compressed_dataset = file.read()
        dumped_dataset = zlib.decompress(compressed_dataset)
        dataset = pickle.loads(dumped_dataset)
    return dataset


def save_binary_dataset(dataset: BinaryDataset, ext: str = 'bin') -> None:
    dataset_path = get_dataset_path(dataset.dataset_name, ext=ext)
    with open(dataset_path, 'wb') as file:
        dumped_dataset = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_dataset = zlib.compress(dumped_dataset)
        file.write(compressed_dataset)


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
    langs = LANGS[chosen_dataset_name]
    for langs_pair in langs:
        if not langs_pair:
            continue
        source_dataset = DatasetFactory.get(chosen_dataset_name, langs_pair)
        source_dataset = DatasetFactory.crop(source_dataset, chosen_dataset_name, size)
        if source_dataset:
            collection.append(source_dataset)
    if save:
        for dataset in collection:
            save_binary_dataset(dataset, ext=ext)
    return collection


if __name__ == '__main__':
    main_args = form_args()
    source_datasets = collect(main_args.dataset_name, save=True, ext=main_args.bin_ext)
    for binary_dataset in source_datasets:
        save_binary_dataset(binary_dataset, ext=main_args.bin_ext)
