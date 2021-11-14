from typing import Any, Callable, Dict, List, Optional, Tuple

from detection.models.translation import TranslationModel
from detection.arguments import form_args
from detection.data.factory import DatasetFactory
from detection.data.wrapper import TextDetectionDataset
from detection.utils import get_mock_dataset, log, save, save_translations


DATASET_SIZE = 10000


def extend_translations(
        translations: List[str],
        dataset: Any,
        translate: Callable[[str], str],
        index: int,
        sample: Dict[str, Any]) -> List[str]:
    src_lang, trg_lang = DatasetFactory.get_languages(dataset.dataset_name)
    src, trg = sample[src_lang], sample[trg_lang]
    log(index, len(dataset), src)
    # TODO: правда ли, что эта операция происходит на GPU?
    gen = translate(src)
    translations.extend([gen, trg])
    return translations


def translate_dataset(
        dataset: Any,
        translate: Callable[[str], str],
        dataset_name: Optional[str] = None,
        device: Optional[str] = None,
        ext: str = 'bin'
) -> List[str]:
    translations = []
    for i, sample in enumerate(dataset):
        translations = extend_translations(
            translations, dataset, translate, i, sample
        )
        save(translations, dataset_name, i, len(dataset), device, ext)
    return translations


def get_generation_dataset(dataset: Optional[Any] = None,
                           size: Optional[int] = None) -> Any:
    if not dataset:
        dataset = get_mock_dataset()
    if not size:
        size = len(dataset['train'])
    dataset = dataset['train'][:size]['translation']
    return dataset


def generate(dataset: Optional[Any] = None,
             size: Optional[int] = None,
             device: Optional[str] = None,
             ext: str = 'bin') -> Tuple[TextDetectionDataset, TextDetectionDataset]:
    dataset = get_generation_dataset(dataset, size=size)
    model = TranslationModel(device=device)
    translations = translate_dataset(
        dataset=dataset,
        translate=model,
        dataset_name=dataset.dataset_name,
        device=device,
        ext=ext
    )
    return save_translations(translations, dataset.dataset_name, device, ext)


if __name__ == '__main__':
    # TODO: для DVC должен взаимодействовать с модулем collect()
    main_args = form_args()
    generate(main_args.size)
