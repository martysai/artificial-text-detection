import os.path as path
from unittest import TestCase

import pandas as pd
from hamcrest import assert_that, equal_to


from detection.arguments import form_args
from detection.data.generate_language_model import (
    generate_language_model, retrieve_prefix, parse_collection_on_repeats, super_maximal_repeat, filter_collection
)
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
            "kkkkkkk",
            "абв гдеж абв гдеж",
            "привет я звать привет"
        ]
        cls.smr_targets = [
            "",
            " sent. ",
            "three",
            "kkkkkk",
            "абв гдеж",
            "привет"
        ]
        cls.prefixes = [
            "",
            "One sent. Two sent.",
            "abc xx one. two_three abc abc-three.",
            "kkkkkkk.",
            "абв гдеж абв гдеж.",
            "привет я звать привет."
        ]

        cls.process_df = pd.DataFrame([
            {
                "text": "Доусон также подтвердил, что вопрос об использовании средств МВФ будет обязательно "
                        "рассмотрен во время предстоящего обсуждения итогов работы миссии фонда в Москве, "
                        "которая представит совету директоров МВФ доклад о результатах своих переговоров с "
                        "представителями российского правительства.",
            },
            {
                "text": "Местные телекомпании, ведущие прямые репортажи непосредственно с места катастрофы, сообщают, "
                        "что произошедшая трагедия - крупнейшая в истории гражданской авиации Аргентины.",
            }
        ], columns=["text"])

        cls.repeats = [
            "привет меня зовут.",
            "привет мне",
            "как твои дела? привет"
        ]

        cls.args = form_args()
        cls.unsupervised_baseline = UnsupervisedBaseline(args=cls.args)
        dir_path = path.dirname(path.dirname(path.realpath(__file__)))
        dvc_path = path.join(dir_path, "resources/data")
        cls.news_df = pd.read_csv(path.join(dvc_path, "lenta/lenta_sample.csv"), nrows=20)

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

    def test_parse_collection_on_repeats_mock(self) -> None:
        mock_list = self.process_df["text"].values.tolist()
        repeats = parse_collection_on_repeats(mock_list, collection_length=10, smr_length=4)
        assert_that(len(repeats), equal_to(6))
        assert_that(repeats[3], equal_to(" мвф "))

    def test_parse_collection_on_repeats_sample(self) -> None:
        repeats = parse_collection_on_repeats(self.repeats, collection_length=10, smr_length=4)
        assert_that(len(repeats), equal_to(2))
        assert_that(repeats[0], equal_to("привет м"))

    def test_parse_collection_on_repeats(self) -> None:
        lenta_list = self.news_df["text"].values.tolist()
        repeats = parse_collection_on_repeats(lenta_list)
        assert_that(len(repeats), equal_to(35))

    def test_filter_collection(self) -> None:
        filtered = filter_collection(self.process_df["text"].values.tolist())
        assert_that(len(filtered), equal_to(2))

    def test_label_with_repeats(self) -> None:
        df = pd.DataFrame(self.paragraphs, columns=["text"])
        df = UnsupervisedBaseline.label_with_repeats(df, collection_length=10, smr_length=4, seed=1206)
        assert_that(df["target"].values.tolist(), equal_to(["human", "human", "machine", "human", "human", "machine"]))

    def test_semi_supervise_mock(self) -> None:
        generated_df = UnsupervisedBaseline.process(self.process_df)
        generated_df = UnsupervisedBaseline.semi_supervise(generated_df, supervise_rate=0.5, seed=1206)
        assert_that(generated_df["target"].values.tolist(), equal_to(["human", "human", "human"]))

    def test_semi_supervise(self) -> None:
        pass


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
        generated_texts = self.language_model(self.paragraphs)
        # assert_that(generated_texts[0][42:92], equal_to("Нет, я не умру от боли. Ну пожалуйста, пожалуйста…"))

    def test_language_model_dataset(self) -> None:
        generated_dataset = generate_language_model(self.paragraphs)
        assert_that(len(generated_dataset), equal_to(3))
