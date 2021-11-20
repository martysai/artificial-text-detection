import os.path as path
from typing import Callable, Collection, Dict, List, Optional, Union

from detection.models.translation import TranslationModel
from detection.arguments import form_args, get_dataset_path
from detection.data.factory import BinaryDataset, DatasetFactory, collect, load_binary_dataset
from detection.utils import MockDataset, TrainEvalDatasets, log, save, save_translations, save_translations_texts


def translate_dataset(
        dataset: Collection[Dict[str, str]],
        translate: Callable[[Union[str, List[str]]], Union[str, List[str]]],
        dataset_name: Optional[str] = None
) -> List[str]:
    # TODO: порефакторить все вызовы translate_dataset
    src_lang, trg_lang = DatasetFactory.get_languages(dataset_name)
    sources = [sample[src_lang] for sample in dataset]
    translated = translate(sources)
    return translated


def get_generation_dataset(dataset: BinaryDataset,
                           dataset_name: Optional[str],
                           size: Optional[int] = None) -> Union[Collection[Dict[str, str]], MockDataset]:
    """
    This method prepares a dataset which is put in generate.
    """
    if not dataset:
        return MockDataset()
    if dataset_name == 'tatoeba':
        dataset = dataset['train']['translation']
    elif dataset_name == 'wikimatrix':
        # TODO: transform to list for wikimatrix
        pass
    if size:
        dataset = dataset[:size]
    return dataset


def generate(dataset: BinaryDataset,
             dataset_name: str,
             size: Optional[int] = None,
             device: Optional[str] = None,
             ext: str = 'bin') -> TrainEvalDatasets:
    """
    Parameters
    ----------
        TODO
        dataset: detection.data.factory.BinaryDataset
        dataset_name: str
        size: Optional[int]
        device: Optional[str]
        ext: str

    Returns
    -------
        datasets: TrainEvalDatasets
            A tuple containing train and eval datasets.
    """
    dataset = get_generation_dataset(dataset, dataset_name=dataset_name, size=size)
    # TODO: add the support of languages
    model = TranslationModel()
    translations = translate_dataset(
        dataset=dataset,
        translate=model,
        dataset_name=dataset_name,
    )
    src_lang, trg_lang = DatasetFactory.get_languages(dataset_name)
    sources = [sample[src_lang] for sample in dataset]
    targets = [sample[trg_lang] for sample in dataset]
    save_translations_texts(sources, targets, translations, dataset_name)
    return save_translations(targets, translations, dataset_name, device, ext)


if __name__ == '__main__':
    # TODO: (для DVC) должен взаимодействовать с модулем collect()
    main_args = form_args()
    binary_dataset_path = get_dataset_path(main_args.dataset_name, 'bin')
    if not path.exists(binary_dataset_path):
        datasets = collect(chosen_dataset_name=main_args.dataset_name, save=True, size=main_args.size, ext='bin')
    else:
        binary_dataset = load_binary_dataset(main_args.dataset_name)
        # TODO: improve for other languages
        datasets = [binary_dataset]

    for binary_ind, binary_dataset in enumerate(datasets):
        # TODO: add logs with languages
        print(f'Handling a binary dataset = {binary_ind + 1}')
        train_dataset, eval_dataset = generate(
            dataset=binary_dataset,
            dataset_name=main_args.dataset_name,
            device=main_args.device,
            size=main_args.size,
            ext=main_args.ds_ext
        )
        train_gen_path = get_dataset_path(f'{main_args.dataset_name}.train.{binary_ind + 1}.gen', 'pth')
        train_dataset.save(train_gen_path)
        eval_gen_path = get_dataset_path(f'{main_args.dataset_name}.eval.{binary_ind + 1}.gen', 'pth')
        eval_dataset.save(eval_gen_path)
