from typing import Any, List, Optional, Tuple, Union

from easynmt import EasyNMT

from artificial_detection.models.const import BATCH_SIZE, EASY_NMT_MODEL_NAME, SRC_LANG, TRG_LANG


def handle_bart_langs(src_lang: str, trg_lang: str) -> Tuple[str, str]:
    if (not src_lang) or (not trg_lang):
        raise ValueError("Should pass languages if BART model is chosen")
    src_end = "XX" if src_lang in ["fr", "es", "en"] else src_lang.upper()
    src_lang = f"{src_lang}_{src_end}"
    trg_end = "XX" if trg_lang in ["fr", "es", "en"] else trg_lang.upper()
    trg_lang = f"{trg_lang}_{trg_end}"
    return src_lang, trg_lang


def is_bart(model: Any, model_name: Optional[str] = None) -> bool:
    if model_name:
        return model_name == "mbart"
    if isinstance(model, str):
        return model == "mbart"
    # TODO: improve for EasyNMT object
    return False


def retrieve_model(model: Any, device: str) -> Any:
    if not model:
        return EasyNMT(EASY_NMT_MODEL_NAME, device=device)
    if isinstance(model, str):
        return EasyNMT(model, device=device)
    return model


class TranslationModel:
    """
    Model which is used for inference to generate artificial texts.

    Attributes
    ----------
    model: any
        An instance of EasyNMT model.
    src_lang: str
        Source language for the translation inference.
    trg_lang: str
        Target language for the translation inference.
    batch_size: int
        Batch size for the translation inference.

    Methods
    -------
    __call__(source)
        Able to translate a single text or a list of texts.
    """
    def __init__(
        self,
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        src_lang: Optional[str] = None,
        trg_lang: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = "cpu",
    ) -> None:
        self.model = retrieve_model(model, device)
        self.model_name = model_name
        if is_bart(model, model_name):
            src_lang, trg_lang = handle_bart_langs(src_lang, trg_lang)
        self.src_lang = src_lang or SRC_LANG
        self.trg_lang = trg_lang or TRG_LANG
        self.batch_size = batch_size or BATCH_SIZE

    def __call__(self, source: Union[List[str], str]) -> Union[List[str], str]:
        return self.model.translate(
            source,
            source_lang=self.src_lang,
            target_lang=self.trg_lang,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )
