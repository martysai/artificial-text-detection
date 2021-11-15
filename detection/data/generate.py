import os.path as path
from typing import Any, Callable, Collection, Dict, List, Optional, Union

from detection.models.translation import TranslationModel
from detection.arguments import form_args, get_dataset_path
from detection.data.factory import BinaryDataset, DatasetFactory, collect, load_binary_dataset
from detection.utils import MockDataset, TrainEvalDatasets, log, save, save_translations


DATASET_SIZE = 10000


def extend_translations(
        translations: List[str],
        dataset: Collection[Dict[str, str]],
        dataset_name: str,
        translate: Callable[[str], str],
        index: int,
        sample: Dict[str, Any]) -> List[str]:
    src_lang, trg_lang = DatasetFactory.get_languages(dataset_name)
    src, trg = sample[src_lang], sample[trg_lang]
    log(index, len(dataset), src)
    # TODO: правда ли, что эта операция происходит на GPU?
    gen = translate(src)
    translations.extend([gen, trg])
    return translations


def translate_dataset(
        dataset: Collection[Dict[str, str]],
        translate: Callable[[str], str],
        dataset_name: Optional[str] = None,
        device: Optional[str] = None,
        ext: str = 'bin'
) -> List[str]:
    translations = []
    for i, sample in enumerate(dataset):
        translations = extend_translations(
            translations, dataset, dataset_name, translate, i, sample
        )
        save(translations, dataset_name, i, len(dataset), device, ext)
    return translations


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
        # TODO: extend for wikimatrix
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
    model = TranslationModel(device=device)
    translations = translate_dataset(
        dataset=dataset,
        translate=model,
        dataset_name=dataset_name,
        device=device,
        ext=ext
    )
    return save_translations(translations, dataset_name, device, ext)


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
        # TODO: add logs
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
