from unittest import TestCase
from hamcrest import assert_that, equal_to

from detection.data.generate_LM import retrieve_prefix, super_maximal_repeat
from detection.models.language_model import LanguageModel
from detection.models.smr.core import SuffixArray


class TestSuperMaximalRepeat(TestCase):
    def test_super_maximal_repeat(self) -> None:
        smr_data = [
            ("", ""),
            ("abc xx one two_three abc abc-three", "three"),
            ("kkkkkkk", "kkkkkk"),
        ]

        for paragraph, repeat in smr_data:
            assert_that(super_maximal_repeat(paragraph), equal_to(repeat))


def test_retrieve_prefix():
    pass


def test_deterministic_output():
    pass
