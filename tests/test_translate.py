from unittest import TestCase

import pandas as pd
from hamcrest import assert_that, equal_to, has_items

from detection.arguments import get_dataset_path
from detection.models.translation import TranslationModel
from detection.utils import save_translations_texts
from tests import skip_github


class TestTranslate(TestCase):
    """
    This test handles EasyNMT functionality implemented via detection.models.translation.TranslationModel
    """
    @classmethod
    def setUpClass(cls) -> None:
        model = TranslationModel()
        sources = [
            'Muiriel ist jetzt 20.',
            'Was ist das?',
            'Das wird nichts ändern.',
            'Muiriel ist jetzt 20.',
            'Ich finde keine Worte.'
        ]
        targets = [
            'Muiriel is 20 now.',
            "What's that?",
            'That will change nothing.',
            'Muiriel has turned twenty.',
            "I don't know what to say."
        ]
        cls.model = model
        cls.sources = sources
        cls.targets = targets
        cls.translations = model(sources)

    def test_translate(self) -> None:
        source = 'Guten Tag.'
        target = self.model(source)
        assert_that(target, equal_to('Hello.'))

    def test_translation_size(self) -> None:
        assert_that(self.translations[0], equal_to('Muiriel is now 20.'))
        assert_that(len(self.translations), equal_to(5))

    def test_save_to_csv(self) -> None:
        dataset_name = 'tatoeba_sample'
        save_translations_texts(self.sources, self.targets, self.translations, dataset_name=dataset_name)
        dataset_path = get_dataset_path(dataset_name, ext='csv')
        df_sample = pd.read_csv(dataset_path)
        assert_that(df_sample.columns.tolist(), has_items(*['sources', 'targets', 'translations']))
        assert_that(len(df_sample), equal_to(5))

    @skip_github
    def test_gpu_usage(self) -> None:
        # TODO
        pass
