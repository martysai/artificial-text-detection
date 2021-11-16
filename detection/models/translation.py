from easynmt import EasyNMT
from typing import List, Optional, Union

EASY_NMT_MODEL_NAME = 'opus-mt'
SRC_LANG = 'ru'
TRG_LANG = 'en'


class TranslationModel:
    def __init__(
            self,
            model=None,
            source_lang=None,
            target_lang=None,
            batch_size=None,
            device=None
    ) -> None:
        self.model = model if model else EasyNMT(EASY_NMT_MODEL_NAME)
        if device and not isinstance(self.model, EasyNMT):
            self.model = self.model.to(device)
        self.source_lang = source_lang or SRC_LANG
        self.target_lang = target_lang or TRG_LANG
        self.batch_size = batch_size
        self.device = device

    def __call__(
            self,
            source: Union[List[str], str],
            device: Optional[str] = None,
            ext: str = 'pth'
    ) -> Union[List[str], str]:
        if not device:
            device = self.device
        return self.model.translate(
            source,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            batch_size=self.batch_size
        )
        # TODO: add cuda supoprt
