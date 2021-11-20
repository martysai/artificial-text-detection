import os
import os.path as path
from typing import List, Union
from unittest import TestCase

import numpy as np
from hamcrest import assert_that, equal_to, has_items
from transformers import EvalPrediction

from detection.data.factory import collect
from detection.data.generate import get_generation_dataset, translate_dataset
from detection.models.validate import compute_metrics
from detection.utils import MockDataset, save_translations


def reverse_transform(s: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(s, str):
        return ''.join(reversed(s))
    return [''.join(reversed(element)) for element in s]


class TestFunctionality(TestCase):
    dataset = None
    translations = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = MockDataset.dataset
        cls.targets = MockDataset.targets()
        cls.translations = translate_dataset(cls.dataset, reverse_transform, dataset_name='mock')

    @classmethod
    def tearDownClass(cls) -> None:
        for suffix in ['train', 'eval']:
            mock_path = path.join(path.dirname(os.getcwd()), f"resources/data/mock.{suffix}.bin")
            if path.exists(mock_path):
                os.remove(mock_path)

    def test_translations(self):
        assert_that(len(self.translations), equal_to(2))
        assert_that(self.translations, equal_to(['gat netug', 'diel rim tut se']))

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
        train_dataset, eval_dataset = save_translations(
            self.targets,
            self.translations,
            dataset_name='mock',
            device='cpu',
            ext='bin'
        )
        encodings_keys = list(train_dataset.encodings.keys())
        assert_that(encodings_keys, has_items(*['input_ids', 'attention_mask']))
        assert_that(len(train_dataset.encodings['input_ids']), equal_to(3))
        assert_that(len(eval_dataset.encodings['input_ids']), equal_to(1))

        for suffix in ['train', 'eval']:
            elder_path = path.join(path.dirname(os.getcwd()), f"resources/data/mock.{suffix}.bin")
            younger_path = path.join(os.getcwd(), f"resources/data/mock.{suffix}.bin")
            assert_that(
                path.exists(elder_path) or path.exists(younger_path),
                equal_to(True)
            )

    def test_generation_dataset(self):
        dataset_name, ext = 'tatoeba', 'bin'
        datasets = collect(dataset_name, save=True, ext=ext)
        dataset = datasets[0]
        sized_dataset = get_generation_dataset(dataset, dataset_name, size=10)
        assert_that(len(sized_dataset), equal_to(10))
        unsized_dataset = get_generation_dataset(dataset, dataset_name)
        assert_that(len(unsized_dataset), equal_to(299769))
