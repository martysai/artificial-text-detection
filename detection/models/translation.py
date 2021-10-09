from easynmt import EasyNMT
from typing import List, Union
from detection.data.generate import TRG_LANG

EASY_NMT_MODEL_NAME = 'opus-mt'


class TranslationModel:
    def __init__(
            self,
            model=None,
            target_lang=None,
    ) -> None:
        self.model = model if model else EasyNMT(EASY_NMT_MODEL_NAME)
        self.target_lang = target_lang if target_lang else TRG_LANG

    def __call__(
            self,
            source: Union[List[str], str]
    ) -> Union[List[str], str]:
        return self.model.translate(
            source,
            target_lang=self.target_lang
        )
