from typing import List, Union

from easynmt import EasyNMT

BATCH_SIZE = 128
EASY_NMT_MODEL_NAME = 'opus-mt'
SRC_LANG = 'de'
TRG_LANG = 'en'


class TranslationModel:
    def __init__(
            self,
            model=None,
            src_lang=None,
            trg_lang=None,
            batch_size=None,
    ) -> None:
        self.model = model if model else EasyNMT(EASY_NMT_MODEL_NAME)
        self.src_lang = src_lang or SRC_LANG
        self.trg_lang = trg_lang or TRG_LANG
        self.batch_size = batch_size or BATCH_SIZE

    def __call__(
            self,
            source: Union[List[str], str]
    ) -> Union[List[str], str]:
        return self.model.translate(
            source,
            source_lang=self.src_lang,
            target_lang=self.trg_lang,
            batch_size=self.batch_size,
            show_progress_bar=True
        )
