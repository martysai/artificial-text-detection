import pickle
import zlib
from collections import defaultdict
from typing import Any, List, Optional

from datasets import load_dataset

from detection.arguments import form_args, get_dataset_path

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

LANGS = {
    'tatoeba': [
        ['en', 'ru'],
        [],
    ],
    'wikimatrix': [
        [],
    ]
}


class DatasetFactory:
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
        for langs_pair in LANGS[dataset_name]:
            if langs_pair:
                return langs_pair
        return None


def load_binary_dataset():
    pass


def save_binary_dataset(dataset: Any, ext: str = 'bin') -> None:
    dataset_path = get_dataset_path(dataset.dataset_name, ext=ext)
    with open(dataset_path, 'wb') as file:
        dumped_dataset = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_dataset = zlib.compress(dumped_dataset)
        file.write(compressed_dataset)


def collect(chosen_dataset_name: str, save: bool = False, ext: str = 'bin') -> List[Any]:
    """
    TODO: написать развёрнутый docstring
    """
    collection = []
    langs = LANGS[chosen_dataset_name]
    for langs_pair in langs:
        if not langs_pair:
            continue
        source_dataset = DatasetFactory.get(chosen_dataset_name, langs_pair)
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
