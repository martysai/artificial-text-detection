from typing import List, Optional, Union

from easynmt import EasyNMT

BATCH_SIZE = 128
SRC_LANG = "ru"
TRG_LANG = "en"


class TranslationModel:
    def __init__(
        self,
        model: Optional[str] = None,
        src_lang: str = None,
        trg_lang: str = None,
        batch_size: int = None,
        is_bart: bool = False,
    ) -> None:
        self.model = model if model else EasyNMT(EASY_NMT_MODEL_NAME)
        self.is_bart = is_bart
        if self.is_bart:
            if (not src_lang) or (not trg_lang):
                raise ValueError("Should pass languages if BART model is chosen")
            src_end = "XX" if src_lang in ["fr", "es", "en"] else src_lang.upper()
            src_lang = f"{src_lang}_{src_end}"
            trg_end = "XX" if trg_lang in ["fr", "es", "en"] else trg_lang.upper()
            trg_lang = f"{trg_lang}_{trg_end}"
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
