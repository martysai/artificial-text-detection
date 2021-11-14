import os.path as path
from unittest import TestCase

import numpy as np
from hamcrest import assert_that, equal_to
from transformers import EvalPrediction

from detection.data.factory import collect
from detection.data.generate import get_generation_dataset, translate_dataset
from detection.models.validate import compute_metrics
from detection.utils import get_mock_dataset, save_translations


def reverse_transform(s: str) -> str:
    return ''.join(reversed(s))


class TestFunctionality(TestCase):
    dataset = None
    translations = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = get_mock_dataset()

        cls.translations = translate_dataset(cls.dataset, reverse_transform)

    def test_translations(self):
        assert_that(len(self.translations), equal_to(4))
        assert_that(self.translations[0], equal_to('речев йырбод'))

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

    def test_save_translations(self):
        train_dataset, eval_dataset = save_translations(self.translations, dataset_name='mock', device='cpu', ext='bin')
        assert_that(len(train_dataset.encodings), equal_to(2))
        assert_that(len(eval_dataset.encodings), equal_to(2))

        # TODO: add path.exists

    def test_generation_dataset(self):
        dataset_name, ext = 'tatoeba', 'bin'
        datasets = collect(dataset_name, save=True, ext=ext)
        dataset = datasets[0]
        sized_dataset = get_generation_dataset(dataset, size=10)
        assert_that(len(sized_dataset), equal_to(10))
        unsized_dataset = get_generation_dataset(dataset)
        assert_that(len(unsized_dataset), equal_to(514195))
