from typing import Any, Dict, List, Optional

from detection.models.const import (
    COLLECTION_CONCAT_SYMBOL,
    LM_LENGTH_LOWER_BOUND,
    LM_LENGTH_UPPER_BOUND,
    SMR_LENGTH_LOWER_BOUND
)
from detection.models.language_model import LanguageModel
from detection.models.smr.core import SuffixArray


def check_input_paragraph(paragraph: str) -> bool:
    """
    TODO: понять, где вызывать предикат
    """
    return len(paragraph) >= LM_LENGTH_LOWER_BOUND


def check_output_paragraph(paragraph: str) -> bool:
    return len(paragraph) >= LM_LENGTH_UPPER_BOUND


def trim_output_paragraph(paragraph: str) -> str:
    return paragraph[:LM_LENGTH_UPPER_BOUND]


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


def parse_collection_on_repeats(collection: List[str]) -> List[str]:
    collection_concat = COLLECTION_CONCAT_SYMBOL.join(collection)
    current_repeat = collection_concat
    repeats = []
    while len(current_repeat) > SMR_LENGTH_LOWER_BOUND and len(collection_concat) > LM_LENGTH_LOWER_BOUND:
        current_repeat = super_maximal_repeat(collection_concat)
        collection_concat = collection_concat.replace(current_repeat, "").replace("  ", " ").strip()
        repeats.append(current_repeat)
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
