from hamcrest import assert_that, equal_to
from unittest import TestCase

from detection.models.translation import TranslationModel


class TestTranslate(TestCase):
    def test_translate(self):
        translation_model = TranslationModel()
        source = 'добрый день'
        target = translation_model(source)
        assert_that(target, equal_to('Good afternoon.'))
