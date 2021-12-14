from typing import Any, Dict, List, Optional

from detection.models.smr.core import SuffixArray
from detection.models.language_model import LanguageModel


def check_paragraph(paragraph: str) -> bool:
    """
    TODO: понять, где вызывать предикат
    """
    return not len(paragraph) < 100


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


def generate_LM(
    paragraphs: List[str],
    sentence_num: int = 2,
    lm_params: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    TODO-Doc
    """
    generated_paragraphs = []
    language_model = LanguageModel()

    for paragraph in paragraphs:
        prefix = retrieve_prefix(paragraph, sentence_num=sentence_num)
        generated_paragraphs.append(language_model(prefix, **lm_params))
    return generated_paragraphs
