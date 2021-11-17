from hamcrest import assert_that, equal_to
from unittest import TestCase

from detection.models.translation import TranslationModel
from tests import skip_github


class TestTranslate(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = TranslationModel()

    def test_translate(self):
        source = 'добрый день'
        target = self.model(source)
        assert_that(target, equal_to('Good afternoon.'))

    def test_translation_size(self):
        src_corpus = [
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
        translations = self.model(src_corpus)
        assert_that(translations[0], equal_to("Once in my life, I do a good thing... and it's useless."))
        assert_that(len(translations), equal_to(10))

    def test_save_to_csv(self):
        # TODO: add testing that df has been saved
        pass

    @skip_github
    def test_gpu_usage(self):
        # TODO
        pass
