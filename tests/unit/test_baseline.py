from hamcrest import assert_that, equal_to
from unittest import TestCase

from detection.data.generate import get_buffer


class TestBaseline(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        pass

    def test_buffer(self):
        dataset = ...  # TODO: arrow_dataset.DataSet
        transform = reversed
        buffer = get_buffer(dataset, transform)

        # TODO: fulfil equal_to
        assert_that(len(buffer), equal_to(...))
        assert_that(buffer[0], equal_to(...))
