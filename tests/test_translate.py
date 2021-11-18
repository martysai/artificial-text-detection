from unittest import TestCase

import pandas as pd
from hamcrest import assert_that, equal_to, has_items

from detection.arguments import get_dataset_path
from detection.models.translation import TranslationModel
from detection.utils import save_translations_texts
from tests import skip_github


class TestTranslate(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model = TranslationModel()
        sources = [
            'Один раз в жизни я делаю хорошее дело... И оно бесполезно.',
            'Давайте что-нибудь попробуем!',
            'Давайте что-нибудь попробуем.',
            'Мне пора идти спать.',
            'Мне нужно идти спать.',
            'Я должен ложиться спать.',
            'Сегодня 18 июня, и это день рождения Мюриэл!',
            'Мюриэл сейчас 20.',
            'Пароль «Muiriel».',
            'Я скоро вернусь.'
        ]
        cls.model = model
        cls.sources = sources
        cls.translations = model(sources)

    def test_translate(self) -> None:
        source = 'добрый день'
        target = self.model(source)
        assert_that(target, equal_to('Good afternoon.'))

    def test_translation_size(self) -> None:
        assert_that(self.translations[0], equal_to("Once in my life, I do a good thing... and it's useless."))
        assert_that(len(self.translations), equal_to(10))

    def test_save_to_csv(self) -> None:
        dataset_name = 'tatoeba_sample'
        save_translations_texts(self.sources, self.translations, dataset_name=dataset_name)
        dataset_path = get_dataset_path(dataset_name, ext='csv')
        df_sample = pd.read_csv(dataset_path)
        assert_that(df_sample.columns.tolist(), has_items(*['sources', 'translations']))
        assert_that(len(df_sample), equal_to(10))

    @skip_github
    def test_gpu_usage(self) -> None:
        # TODO
        pass
