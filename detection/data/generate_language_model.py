from tqdm import tqdm
from typing import Any, Dict, List, Optional

from detection.models.const import (
    COLLECTION_CONCAT_SYMBOL,
    LM_LENGTH_LOWER_BOUND,
    LM_LENGTH_UPPER_BOUND,
    ORD_UPPER_BOUND,
    SMR_LENGTH_LOWER_BOUND
)
from detection.models.language_model import LanguageModel
from detection.models.smr.core import SuffixArray
from detection.utils import ord_cyrillic
from string import punctuation


def check_input_paragraph(paragraph: str) -> bool:
    """
    TODO: понять, где вызывать предикат
    """
    return len(paragraph) >= LM_LENGTH_LOWER_BOUND


def check_output_paragraph(paragraph: str) -> bool:
    return len(paragraph) >= LM_LENGTH_UPPER_BOUND


def trim_output_paragraph(paragraph: str) -> str:
    trimmed = paragraph[:LM_LENGTH_UPPER_BOUND]
    if trimmed.rfind(".") > LM_LENGTH_LOWER_BOUND:
        return trimmed[:trimmed.rfind(".") + 1]
    return trimmed


def retrieve_prefix(paragraph: str, sentence_num: int = 2) -> str:
    """
    TODO-Doc
    """
    sentences = paragraph.strip().split('.')
    sentences = [sent.strip() + '.' for sent in sentences if len(sent)]
    prefix = " ".join(sentences[:sentence_num])
    return prefix


def super_maximal_repeat(paragraph: str) -> str:
    suffix_array = SuffixArray(paragraph)
    return suffix_array.longest_repeated_substring()


def preprocess_text(text: str) -> str:
    tokens = text.lower().replace("$", "").split()
    tokens = [
        token for token in tokens if token != " " and token.strip() not in punctuation
    ]
    text = " ".join(tokens)
    return text


def filter_collection(collection: List[str]) -> List[str]:
    filtered = []
    for text in collection:
        filtered_text = preprocess_text(text)
        filtered_text = ''.join(list(filter(lambda c: ord_cyrillic(c) < ORD_UPPER_BOUND, filtered_text)))
        filtered.append(filtered_text)
    return filtered


def parse_collection_on_repeats(
    collection: List[str],
    collection_length: int = LM_LENGTH_LOWER_BOUND,
    smr_length: int = SMR_LENGTH_LOWER_BOUND
) -> List[str]:
    """
    Collecting all super maximal repeats from a set of documents.
    Collection is trivially preprocessed with eliminating punctuation.

    # TODO: сохранять repeats
    """
    collection = filter_collection(collection)
    collection_concat = COLLECTION_CONCAT_SYMBOL.join(collection)
    pbar = tqdm(total=len(collection_concat) - LM_LENGTH_LOWER_BOUND)
    prev_concat, current_repeat, repeats = "", collection_concat, []
    while prev_concat != collection_concat and len(collection_concat) > collection_length \
            and len(current_repeat) > smr_length:
        current_repeat = super_maximal_repeat(collection_concat)
        prev_concat = collection_concat

        # Check if SMR was false
        if collection_concat.count(current_repeat) >= 2:
            repeats.append(current_repeat)

        collection_concat = collection_concat.replace(current_repeat, "").replace("  ", " ").strip()
        pbar.update(len(current_repeat))
    return repeats


def generate_language_model(
        paragraphs: List[str],
        sentence_num: int = 2,
        lm_params: Optional[Dict[str, Any]] = None,
        size: Optional[int] = None
) -> List[str]:
    """
    TODO-Doc
    """
    language_model = LanguageModel()
    prefixes = [
        retrieve_prefix(paragraph, sentence_num=sentence_num)
        for paragraph in paragraphs
    ]
    return language_model(prefixes, **(lm_params or {}))
