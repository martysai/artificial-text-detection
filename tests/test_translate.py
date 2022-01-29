import os
from unittest import TestCase

import pandas as pd
from hamcrest import assert_that, equal_to, has_items

from artificial_detection.models.translation import TranslationModel
from artificial_detection.utils import get_dataset_path, save_translations_texts
from tests import skip_github


class TestTranslate(TestCase):
    """
    This test handles EasyNMT functionality implemented via detection.models.translation.TranslationModel
    """

    @classmethod
    def setUpClass(cls) -> None:
        model = TranslationModel(src_lang="ru", trg_lang="en")
        sources = [
            "Один раз в жизни я делаю хорошее дело... И оно бесполезно.",
            "Давайте что-нибудь попробуем!",
            "Давайте что-нибудь попробуем.",
            "Мне пора идти спать.",
            "Мне нужно идти спать.",
        ]
        targets = [
            "For once in my life I'm doing a good deed... And it is useless.",
            "Let's try something.",
            "Let's try something.",
            "I have to go to sleep.",
            "I have to go to sleep.",
        ]
        cls.model = model
        cls.sources = sources
        cls.targets = targets
        cls.translations = model(sources)

    def test_translate(self) -> None:
        source = "Добрый день."
        target = self.model(source)
        assert_that(target, equal_to("Good afternoon."))

    def test_translation_size(self) -> None:
        assert_that(self.translations[0], equal_to("Once in my life, I do a good thing... and it's useless."))
        assert_that(len(self.translations), equal_to(5))

    def test_save_to_csv(self) -> None:
        dataset_name = "tatoeba_sample"
        save_translations_texts(
            self.sources, self.targets, self.translations, dataset_name=dataset_name, src_lang="ru", trg_lang="en"
        )
        dataset_path = get_dataset_path(f"{dataset_name}.ru-en", ext="csv")
        df_sample = pd.read_csv(dataset_path)
        assert_that(df_sample.columns.tolist(), has_items(*["sources", "targets", "translations"]))
        assert_that(len(df_sample), equal_to(5))
        os.remove(dataset_path)

    @skip_github
    def test_gpu_usage(self) -> None:
        # TODO
        pass
