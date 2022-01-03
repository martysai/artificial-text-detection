from typing import Any, Dict, List, Optional

from collections import defaultdict

import pandas as pd
from datasets import load_dataset

from detection.arguments import form_args, get_dataset_path
from detection.data.data import BinaryDataset
from detection.utils import save_binary_dataset

# --- Datasets configs description ---

CONFIGS = {
    "tatoeba": {
        "path": "tatoeba",
    },
}
CONFIGS = defaultdict(dict, CONFIGS)


# --- Datasets sources description ---


def process_source_wikimatrix(path: str) -> List[str]:
    with open(path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def load_wikimatrix(lang1: str, lang2: str) -> List[Dict[str, str]]:
    sources_path = get_dataset_path(f"wikimatrix/WikiMatrix.{lang1}-{lang2}.txt", ext=lang1)
    sources = process_source_wikimatrix(sources_path)
    targets_path = get_dataset_path(f"wikimatrix/WikiMatrix.{lang1}-{lang2}.txt", ext=lang2)
    targets = process_source_wikimatrix(targets_path)
    dataset = [{lang1: sources[i], lang2: targets[i]} for i in list(range(len(sources)))]
    return dataset


def load_rnc(lang1: str, lang2: str) -> List[Dict[str, str]]:
    sources_path = get_dataset_path("rnc/rnc", langs=[lang1, lang2], ext="txt")
    with open(sources_path) as file:
        lines = file.readlines()
        sources = [line.rstrip() for line in lines]
    dataset = [{lang1: sources[i], lang2: ""} for i in list(range(len(sources)))]
    return dataset


def load_prozhito(lang1: str, lang2: str) -> List[Dict[str, str]]:
    sources_path = get_dataset_path("prozhito/prozhito", langs=[lang1, lang2], ext="csv")
    sources_df = pd.read_csv(sources_path)
    sources = sources_df["sources"].values.tolist()
    dataset = [{lang2: sources[i], lang1: ""} for i in list(range(len(sources)))]
    print("dataset[0]:", dataset[0])
    return dataset


def load_med(lang1: str, lang2: str) -> List[Dict[str, str]]:
    sources_path = get_dataset_path("med/med", langs=[lang1, lang2], ext="csv")
    sources_df = pd.read_csv(sources_path)
    sources = sources_df["sentence"].values.tolist()
    dataset = [{lang1: sources[i], lang2: ""} for i in list(range(len(sources)))]
    return dataset


def load_wiki(lang1: str, lang2: str) -> List[Dict[str, str]]:
    sources_path = get_dataset_path("wiki/wiki", langs=[lang1, lang2], ext="csv")
    sources_df = pd.read_csv(sources_path)
    sources = sources_df["sents"].values.tolist()
    dataset = [{lang1: sources[i], lang2: ""} for i in list(range(len(sources)))]
    return dataset


def load_news(lang1: str, lang2: str) -> List[Dict[str, str]]:
    sources_path = get_dataset_path("news/news", langs=[lang1, lang2], ext="csv")
    sources_df = pd.read_csv(sources_path)
    sources = sources_df["text"].values.tolist()
    dataset = [{lang1: sources[i], lang2: ""} for i in list(range(len(sources)))]
    return dataset


def load_back(lang1: str, lang2: str) -> List[Dict[str, str]]:
    sources_path = get_dataset_path("back/back", langs=[lang1, lang2], ext="csv")
    sources_df = pd.read_csv(sources_path)
    sources = sources_df["forward_translations"].values.tolist()
    dataset = [{lang1: sources[i], lang2: ""} for i in list(range(len(sources)))]
    return dataset


ENTRYPOINTS = {
    "tatoeba": load_dataset,
    "wikimatrix": load_wikimatrix,
    "rnc": load_rnc,
    "prozhito": load_prozhito,
    "med": load_med,
    "wiki": load_wiki,
    "news": load_news,
    "back": load_back,
}
SUPPORTED_DATASETS = list(ENTRYPOINTS.keys())

# --- Using languages description ---
# We suppose that languages follow the order: [source language, target language]
# Every dataset has a default language pair (it should be put first).
# Then, generate.py should not support iterating through several languages.
# This logic is put directly into scripts.
# The third argument can take two possible values: ['straight', 'reversed'].

DEFAULT_LANGS = ["en", "ru", "straight"]
LANGS = {
    "tatoeba": [
        ["en", "ru", "reversed"],
        ["es", "ru", "reversed"],
        ["fi", "ru", "reversed"],
        ["fr", "ru", "reversed"],
    ],
    "wikimatrix": [
        # ["ru", "en", "straight"],
        # ["es", "ru", "reversed"],
        ["fi", "ru", "reversed"],
        # ["fr", "ru", "reversed"],
    ],
    "rnc": [
        # ["ru", "en", "straight"],
        # ["ru", "es", "straight"],
        # ["ru", "fi", "straight"],
        ["ru", "fr", "straight"],
    ],
    "prozhito": [
        # ["en", "ru", "reversed"],
        # ["es", "ru", "reversed"],
        # ["fi", "ru", "reversed"],
        ["fr", "ru", "reversed"],
    ],
    "med": [
        ["ru", "en", "straight"],
        ["ru", "es", "straight"],
        ["ru", "fi", "straight"],
        ["ru", "fr", "straight"],
    ],
    "wiki": [
        # ["ru", "en", "straight"],
        # ["ru", "es", "straight"],
        # ["ru", "fi", "straight"],
        ["ru", "fr", "straight"],
    ],
    "news": [
        # ["ru", "en", "straight"],
        # ["ru", "es", "straight"],
        # ["ru", "fi", "straight"],
        ["ru", "fr", "straight"],
    ],
    "back": [
        ["en", "ru", "straight"],
        ["es", "ru", "straight"],
        ["fi", "ru", "straight"],
        ["fr", "ru", "straight"],
    ]
}
LANGS = defaultdict(list, LANGS)


class DatasetFactory:
    @staticmethod
    def crop(dataset: BinaryDataset, dataset_name: str, size: Optional[int]) -> BinaryDataset:
        if not size:
            return dataset
        if dataset_name == "tatoeba":
            # TODO-Extra: Figure out how to crop better
            # dataset['train']['translation'] = dataset['train']['translation'][:size]
            pass
        if dataset_name != "tatoeba":
            dataset = dataset[:size]
        return dataset

    @staticmethod
    def get(dataset_name: str, langs: List[str]) -> Any:
        config = CONFIGS[dataset_name]
        if dataset_name in SUPPORTED_DATASETS:
            config["lang1"], config["lang2"] = langs[0], langs[1]
        entrypoint = ENTRYPOINTS[dataset_name]
        source_dataset = entrypoint(**config)
        return source_dataset

    @staticmethod
    def get_languages(dataset_name: str) -> List[List[str]]:
        return LANGS[dataset_name]


def collect(
    chosen_dataset_name: str, save: bool = False, size: Optional[int] = None, ext: str = "bin"
) -> List[BinaryDataset]:
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
        raise ValueError("Wrong chosen dataset name")

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


if __name__ == "__main__":
    main_args = form_args()
    source_datasets = collect(main_args.dataset_name, save=True, ext=main_args.bin_ext)
    print("Source datasets length =", len(source_datasets))
