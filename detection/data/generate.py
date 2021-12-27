from typing import Callable, Collection, Dict, List, Optional, Union

import os.path as path

from easynmt import EasyNMT, models

from detection.arguments import form_args, get_dataset_path
from detection.data.factory import SUPPORTED_DATASETS, BinaryDataset, DatasetFactory, collect
from detection.data.wrapper import TextDetectionDataset
from detection.models.translation import TranslationModel
from detection.utils import MockDataset, load_binary_dataset, save_translations_texts, translations_to_torch_dataset


def translate_dataset(
    dataset: Collection[Dict[str, str]],
    translate: Callable[[Union[str, List[str]]], Union[str, List[str]]],
    src_lang: str,
) -> List[str]:
    sources = [sample[src_lang] for sample in dataset]
    translated = translate(sources)
    return translated


def get_generation_dataset(
    dataset: BinaryDataset, dataset_name: Optional[str], size: Optional[int] = None
) -> Union[Collection[Dict[str, str]], MockDataset]:
    """
    This method prepares a dataset which is put in generate.
    """
    if not dataset:
        return MockDataset()
    if dataset_name == "tatoeba":
        dataset = dataset["train"]["translation"]
    elif dataset_name == "wikimatrix":
        # TODO-WikiMatrix
        pass
    if size:
        dataset = dataset[:size]
    return dataset


def generate(
    dataset: BinaryDataset,
    dataset_name: str,
    src_lang: Optional[str] = None,
    trg_lang: Optional[str] = None,
    size: Optional[int] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    easy_nmt_offline: Optional[bool] = None,
    offline_prefix: Optional[str] = None,
    offline_cache_prefix: Optional[str] = None,
    multilingual: bool = False,
    is_bart: bool = False
) -> TextDetectionDataset:
    """
    Generating mappings (sources, targets, translations) for a fixed pair of languages.

    Parameters
    ----------
        dataset: BinaryDataset
            Default source dataset which is processed with factory.
        dataset_name: str
            Possible options: ['tatoeba'].
        src_lang: Optional[str]
            Source language (default is 'ru').
        trg_lang: Optional[str]
            Target language (default is 'en').
        size: Optional[int]
            Size of binary/generated dataset.
        device: Optional[str]
            Where to put torch-like datasets.
        batch_size: Optional[int]
            Batch size for EasyNMT.
        easy_nmt_offline: Optional[bool]
            Flag showing should we use online or offline mode for machine translation.
            See https://github.com/UKPLab/EasyNMT/issues/52
        offline_prefix: Optional[str]
            Where the model is stored (download it with huggingface and git lfs).
        offline_cache_prefix: Optional[str]
            Where the cache is put.
        multilingual: bool
            If True, suppose that model is multilingual.
        is_bart: bool
            If True, suppose that we use BART.

    Returns
    -------
        dataset: TextDetectionDataset
            Torch dataset.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError("Wrong dataset name")

    dataset = get_generation_dataset(dataset, dataset_name=dataset_name, size=size)
    # TODO-EasyNMT: add the support of another EasyNMT

    if not multilingual:
        offline_prefix = f"{offline_prefix}-{src_lang}-{trg_lang}"
    model_config = (
        EasyNMT(translator=models.AutoModel(offline_prefix), cache_folder=offline_cache_prefix)
        if easy_nmt_offline
        else None
    )
    model = TranslationModel(
        model=model_config,
        src_lang=src_lang,
        trg_lang=trg_lang,
        batch_size=batch_size,
        device=device,
        is_bart=is_bart
    )
    translations = translate_dataset(dataset=dataset, translate=model, src_lang=src_lang)
    sources = [sample[src_lang] for sample in dataset]
    targets = [sample[trg_lang] for sample in dataset]
    save_translations_texts(
        sources, targets, translations, dataset_name=dataset_name, src_lang=src_lang, trg_lang=trg_lang
    )
    return translations_to_torch_dataset(targets, translations, device=device, easy_nmt_offline=easy_nmt_offline)


if __name__ == "__main__":
    # TODO-DVC: interact with factory.py
    main_args = form_args()
    languages = DatasetFactory.get_languages(main_args.dataset_name)
    default_binary_dataset_path = get_dataset_path(main_args.dataset_name, langs=languages[0], ext="bin")

    # Retrieving datasets
    if not path.exists(default_binary_dataset_path):
        datasets = collect(
            chosen_dataset_name=main_args.dataset_name, save=True, size=main_args.size, ext=main_args.bin_ext
        )
    else:
        datasets = [
            load_binary_dataset(main_args.dataset_name, langs=lang_pair, ext=main_args.bin_ext)
            for lang_pair in languages
        ]

    # Generating translations and saving torch datasets
    for dataset_ind, (binary_dataset, lang_pair) in enumerate(list(zip(datasets, languages))):
        gl_src_lang, gl_trg_lang, direction = lang_pair
        if direction == "reversed":
            gl_src_lang, gl_trg_lang = gl_trg_lang, gl_src_lang
        elif direction != "straight":
            raise ValueError("Wrong direction passed to language pairs")

        print(
            f"[{dataset_ind + 1}/{len(datasets)}] Handling dataset = {main_args.dataset_name}, "
            f"src lang = {gl_src_lang} trg_lang = {gl_trg_lang}"
        )
        torch_dataset = generate(
            dataset=binary_dataset,
            dataset_name=main_args.dataset_name,
            src_lang=gl_src_lang,
            trg_lang=gl_trg_lang,
            device=main_args.device,
            size=main_args.size,
            batch_size=main_args.easy_nmt_batch_size,
            easy_nmt_offline=main_args.easy_nmt_offline,
            offline_prefix=main_args.offline_prefix,
            offline_cache_prefix=main_args.offline_cache_prefix,
            multilingual=main_args.multilingual,
            is_bart=main_args.is_bart
        )
        train_dataset, eval_dataset = torch_dataset.split()

        train_gen_path = get_dataset_path(
            f"{main_args.dataset_name}.train.{gl_src_lang}-{gl_trg_lang}", ext=main_args.ds_ext
        )
        train_dataset.save(train_gen_path)
        eval_gen_path = get_dataset_path(
            f"{main_args.dataset_name}.eval.{gl_src_lang}-{gl_trg_lang}", ext=main_args.ds_ext
        )
        eval_dataset.save(eval_gen_path)
