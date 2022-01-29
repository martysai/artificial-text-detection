import os
import pickle
import random
import zlib
from os import path
from typing import List, Optional

import pandas as pd
import torch
from transformers import DistilBertTokenizerFast

import wandb
from detection.data.datasets import BinaryDataset, TextDetectionDataset


class MockDataset:
    """
    Mock dataset for testing.
    """
    dataset = [
        {"ru": "добрый день", "en": "good evening",},
        {"ru": "извините", "en": "i am sorry",},
    ]
    _translations = ["good evening", "i am sorry"]
    dataset_name = "mock"

    @classmethod
    def targets(cls) -> List[str]:
        return [sample["en"] for sample in cls.dataset]

    @classmethod
    def translations(cls) -> List[str]:
        return cls._translations

    @classmethod
    def list(cls) -> List[str]:
        dataset_list = []
        for dct in cls.dataset:
            dataset_list.extend([dct["ru"], dct["en"]])
        return dataset_list


def get_dvc_storage_path() -> str:
    """
    Get the full path to the DVC storage.

    Returns
    -------
    str
        Path to the DVC Storage.
    """
    dir_path = path.dirname(path.dirname(path.realpath(__file__)))
    return path.join(dir_path, "resources/data")


def get_dataset_path(dataset_name: str, langs: Optional[List[str]] = None, ext: str = "bin") -> str:
    dvc_path = get_dvc_storage_path()
    if langs:
        dataset_real_name = f"{dataset_name}.{langs[0]}-{langs[1]}.{ext}"
    else:
        dataset_real_name = f"{dataset_name}.{ext}"
    return path.join(dvc_path, dataset_real_name)


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


def ord_cyrillic(c: str) -> int:
    if "а" <= c <= "я":
        return ord(c) - ord("а") + ord("a")  # - cyrillic + latinic
    if "А" <= c <= "Я":
        return ord(c) - ord("А") + ord("A")
    return ord(c)


def setup_experiment_tracking(run_name: str) -> None:
    os.environ["WANDB_MODE"] = "offline"
    token = os.environ.get("WANDB_TOKEN", None)
    wandb.login(key=token)
    wandb.init(project="artificial-text-detection", name=run_name)


def stop_experiment_tracking() -> None:
    wandb.finish()


def fix_random_seed(seed: int = 42) -> None:
    """
    Fixing a random seed.
    """
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
