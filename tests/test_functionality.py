import numpy as np
from unittest import TestCase

from hamcrest import assert_that, equal_to
from transformers import EvalPrediction

from detection.data.generate import get_buffer
from detection.models.validate import compute_metrics
from detection.utils import get_mock_dataset


def reverse_transform(s: str) -> str:
    return ''.join(reversed(s))


class TestFunctionality(TestCase):
    dataset = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = get_mock_dataset()

    def test_buffer(self):
        buffer = get_buffer(self.dataset, reverse_transform)

        assert_that(len(buffer), equal_to(4))
        assert_that(buffer[0], equal_to('речев йырбод'))

    def test_compute_metrics(self):
        eval_pred = EvalPrediction(
            predictions=np.array([[0], [1]]),
            label_ids=np.array([0, 1]),
        )
        results = compute_metrics(eval_pred)
        metrics_names = list(results.keys())
        metrics_values = list(results.values())

        assert_that(metrics_names, equal_to(['accuracy', 'f1', 'precision', 'recall']))
        for value in metrics_values:
            assert_that(type(value), equal_to(float))
