from typing import Callable, List, Tuple, Iterable
from sklearn.model_selection import train_test_split

from datasets import load_dataset

from detection.models.translation import TranslationModel
from detection.data.dataset import TextDetectionDataset

from transformers import DistilBertTokenizerFast


DATASET = 'tatoeba'
SRC_LANG = 'ru'
TRG_LANG = 'en'
TEST_SIZE = 0.2


def get_buffer(
        dataset: Iterable,
        transform: Callable
) -> List[str]:
    buffer = []
    for sample in dataset:
        src, trg = sample[SRC_LANG], sample[TRG_LANG]
        gen = transform(src)
        buffer.extend([gen, trg])
    return buffer


def generate() -> Tuple[TextDetectionDataset, TextDetectionDataset]:
    dataset = load_dataset(DATASET, lang1=SRC_LANG, lang2=TRG_LANG)
    dataset = dataset['train']['translation']
    model = TranslationModel()

    buffer = get_buffer(dataset, model)
    labels = [0, 1] * len(dataset)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        buffer, labels, test_size=TEST_SIZE
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = TextDetectionDataset(train_encodings, train_labels)
    eval_dataset = TextDetectionDataset(eval_encodings, val_labels)

    return train_dataset, eval_dataset
