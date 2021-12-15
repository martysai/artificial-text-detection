from unittest import TestCase
from hamcrest import assert_that, equal_to

from detection.data.generate_LM import retrieve_prefix, super_maximal_repeat
from detection.models.language_model import LanguageModel
from detection.models.smr.core import SuffixArray


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

    def test_super_maximal_repeat(self) -> None:
        for paragraph, repeat in zip(self.paragraphs, self.smr_targets):
            assert_that(super_maximal_repeat(paragraph), equal_to(repeat))

    def test_retrieve_prefix(self) -> None:
        for paragraph, prefix in zip(self.paragraphs, self.prefixes):
            assert_that(retrieve_prefix(paragraph), equal_to(prefix))


def test_deterministic_output():
    pass
