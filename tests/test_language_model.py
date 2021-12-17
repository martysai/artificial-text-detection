from unittest import TestCase

import pandas as pd
from hamcrest import assert_that, equal_to


from detection.data.generate_language_model import generate_language_model, retrieve_prefix, super_maximal_repeat
from detection.models.language_model import LanguageModel
from detection.models.smr.core import SuffixArray
from detection.unsupervised_baseline import UnsupervisedBaseline
from tests import skip_github


class TestUnsupervisedBaselineTools(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paragraphs = [
            "",
            "One sent. Two sent. C. D. E. F.",
            "abc xx one. two_three abc abc-three.",
            "kkkkkkk"
        ]
        cls.smr_targets = [
            "",
            " sent. ",
            "three",
            "kkkkkk"
        ]
        cls.prefixes = [
            "",
            "One sent. Two sent.",
            "abc xx one. two_three abc abc-three.",
            "kkkkkkk."
        ]

        cls.process_df = pd.DataFrame([
            {
                "text": "Доусон также подтвердил, что вопрос об использовании средств МВФ будет обязательно рассмотрен во время предстоящего обсуждения итогов работы миссии фонда в Москве, которая представит совету директоров МВФ доклад о результатах своих переговоров с представителями российского правительства.",
            },
            {
                "text": "Местные телекомпании, ведущие прямые репортажи непосредственно с места катастрофы, сообщают, что произошедшая трагедия - крупнейшая в истории гражданской авиации Аргентины.",
            }
        ], columns=["text"])
        cls.unsupervised_baseline = UnsupervisedBaseline()

    def test_super_maximal_repeat(self) -> None:
        for paragraph, repeat in zip(self.paragraphs, self.smr_targets):
            assert_that(super_maximal_repeat(paragraph), equal_to(repeat))

    def test_retrieve_prefix(self) -> None:
        for paragraph, prefix in zip(self.paragraphs, self.prefixes):
            assert_that(retrieve_prefix(paragraph), equal_to(prefix))

    def test_process(self) -> None:
        # TODO: добавить в тест сами значения, когда появится детерминированность в моделях
        generated_df = UnsupervisedBaseline.process(self.process_df)
        assert_that(len(generated_df), equal_to(4))
        assert_that(generated_df["target"].iloc[0], equal_to("machine"))
        assert_that(generated_df["target"].iloc[1], equal_to("human"))


class TestLanguageModels(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.language_model = LanguageModel()
        cls.paragraphs = [
            "Дайте мне белые крылья. Я утопаю в омуте.",
            "Позитивная мотивация - явно не мой конёк. И мы все умрём.",
            "Дайте пройти, кидайте в шляпу. Кредитные карты, дукаты, злоты."
        ]

    @skip_github
    def test_deterministic_generation(self) -> None:
        # TODO: сделать предсказания детерминированными.
        generated_text = self.language_model(self.paragraphs[0])
        assert_that(generated_text[42:92], equal_to("Нет, я не умру от боли. Ну пожалуйста, пожалуйста…"))

    def test_language_model_dataset(self) -> None:
        generated_dataset = generate_language_model(self.paragraphs)
        assert_that(len(generated_dataset), equal_to(3))
