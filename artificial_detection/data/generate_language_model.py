from string import punctuation
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from artificial_detection.models.const import (
    COLLECTION_CONCAT_SYMBOL,
    LM_LENGTH_LOWER_BOUND,
    LM_LENGTH_UPPER_BOUND,
    ORD_UPPER_BOUND,
    SMR_LENGTH_LOWER_BOUND,
)
from artificial_detection.models.language_model import LanguageModel
from artificial_detection.models.smr.core import SuffixArray
from artificial_detection.utils import ord_cyrillic


def check_input_paragraph(paragraph: str, lower_bound: int = None) -> bool:
    """
    Checkint that the paragraph satisfy to its minimal length.

    Parameters
    ----------
    paragraph: str
        Input paragraph.
    lower_bound: int
        Minimal possible length.

    Returns
    -------
    bool
        Flag signalizes if the paragraph satisfies to its minimal length.
    """
    return len(paragraph) >= (lower_bound or LM_LENGTH_LOWER_BOUND)


def check_output_paragraph(paragraph: str, lower_bound: Optional[int] = None) -> bool:
    """
    Checkint that the output paragraph satisfy to its (larger) minimal length.

    Parameters
    ----------
    paragraph: str
        Output paragraph.
    lower_bound: int
        Minimal possible length.

    Returns
    -------
    bool
        Flag signalizes if the paragraph satisfies to its (larger) minimal length.
    """
    return len(paragraph) >= (lower_bound or LM_LENGTH_UPPER_BOUND)


def trim_output_paragraph(paragraph: str) -> str:
    """
    Trimming the paragraph for common batch sizes during training.

    Parameters
    ----------
    paragraph: str
        Input paragraph.

    Returns
    -------
    str
        Trimmed paragraph.
    """
    trimmed = paragraph[:LM_LENGTH_UPPER_BOUND]
    if trimmed.rfind(".") > LM_LENGTH_LOWER_BOUND:
        return trimmed[:trimmed.rfind(".") + 1]
    return trimmed


def retrieve_prefix(paragraph: str, is_sentence: bool = True, cut_num: int = 1) -> str:
    """
    Get a prefix from the input paragraph.

    Parameters
    ----------
    paragraph: str
        Input paragraph.
    is_sentence: bool
        Retrieving sentences or tokens.
    cut_num: int
        Number of sentences/tokens to retrieve from the text (default is 1).

    Example
    -------
    Simple use case.
    .. code-block:: python
        >>> paragraph = "Good evening. I'm from Russia."
        >>> prefix = retrieve_prefix(paragraph, is_sentence=True, cut_num=1)
        >>> print(prefix)
        "Good evening."

    Returns
    -------
    str
        Retrieved prefix.
    """
    # TODO: Improve for ? and !
    if is_sentence:
        sentences = paragraph.strip().split('.')
        sentences = [sent.strip() + '.' for sent in sentences if len(sent)]
        prefix = " ".join(sentences[:cut_num])
    else:
        tokens = paragraph.strip().split()
        prefix = " ".join(tokens[:cut_num])
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

    Parameters
    ----------
    collection: list of str
        TODO
    collection_length: int
        Default is LM_LENGTH_LOWER_BOUND.
    smr_length: int
        Default is SMR_LENGTH_LOWER_BOUND.

    Returns
    -------
    TODO
    """
    collection = filter_collection(collection)
    collection_concat = COLLECTION_CONCAT_SYMBOL.join(collection)
    pbar = tqdm(total=len(collection_concat) - collection_length)
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
        model_name: Optional[str] = None,
        lm_params: Optional[Dict[str, Any]] = None,
        is_sentence: bool = True,
        cut_num: int = 1,
) -> List[str]:
    """
    Generating paragraphs with a language model by a given prompt.

    Parameters
    ----------
    paragraphs: list of str
        Text paragraphs to be processed in the language model.
    model_name: str
        Model name for the huggingface pipeline. Default is None which means we load sberbank-ai gpt-2 model.
    lm_params: dict from str to any
        Additional language model parameters.
    is_sentence: bool
        Flag defining that we split paragraphs by sentences or by tokens. Default is True.
    cut_num: int
        Number of sentences/tokens depending on is_sentence for paragraphs prefixes retrieval.

    Returns
    -------
    List of str
        List of machine-written paragraphs for a given prompt.
    """
    language_model = LanguageModel()
    prefixes = [
        retrieve_prefix(paragraph, is_sentence=is_sentence, cut_num=cut_num)
        for paragraph in paragraphs
    ]
    return language_model(prefixes, **(lm_params or {}))
