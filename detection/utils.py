from typing import Any, List, Optional, Union

import pickle
import zlib

import pandas as pd
import torch
from transformers import DistilBertTokenizerFast

from detection.arguments import get_dataset_path
from detection.data.datasets import BinaryDataset, TextDetectionDataset

SRC_LANG = "de"
TRG_LANG = "en"


class MockDataset:
    dataset = [
        {
            SRC_LANG: "guten tag",
            TRG_LANG: "good evening",
        },
        {
            SRC_LANG: "es tut mir leid",
            TRG_LANG: "i am sorry",
        },
    ]
    _translations = ["good evening", "i am sorry"]
    dataset_name = "mock"

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


def load_binary_dataset(dataset_name: str, langs: Optional[List[str]] = None, ext: str = "bin") -> BinaryDataset:
    dataset_path = get_dataset_path(dataset_name, langs=langs, ext=ext)
    with open(dataset_path, "rb") as file:
        compressed_dataset = file.read()
        dumped_dataset = zlib.decompress(compressed_dataset)
        dataset = pickle.loads(dumped_dataset)
    return dataset


def save_binary_dataset(
    dataset: BinaryDataset, dataset_name: str, langs: Optional[List[str]] = None, ext: str = "bin"
) -> None:
    dataset_path = get_dataset_path(dataset_name, langs=langs, ext=ext)
    with open(dataset_path, "wb") as file:
        dumped_dataset = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_dataset = zlib.compress(dumped_dataset)
        file.write(compressed_dataset)


def translations_to_torch_dataset(
    targets: List[str], translations: List[str], easy_nmt_offline: Optional[bool] = None, device: Optional[str] = None
) -> TextDetectionDataset:
    corpus = TextDetectionDataset.get_corpus(targets, translations)
    labels = torch.FloatTensor([0, 1] * len(targets))

    tokenizer_path = "resources/data/tokenizer" if easy_nmt_offline else "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)

    encodings = tokenizer(corpus, truncation=True, padding=True)

    encodings, labels = TextDetectionDataset.to_device(encodings, labels, device=device)
    dataset = TextDetectionDataset(encodings, labels, device=device)
    return dataset


def save_translations_texts(
    sources: List[str], targets: List[str], translations: List[str], dataset_name: str, src_lang: str, trg_lang: str
) -> None:
    """
    Saves data to csv.
    """
    print("Saving sources/translations in csv...")
    df_data = list(zip(sources, targets, translations))
    df = pd.DataFrame(data=df_data, columns=["sources", "targets", "translations"])
    csv_path = get_dataset_path(f"{dataset_name}.{src_lang}-{trg_lang}", ext="csv")
    df.to_csv(csv_path, index=False)
