import os.path as path
from unittest import TestCase

from hamcrest import assert_that, equal_to

from detection.arguments import get_dataset_path
from detection.data.factory import collect
from detection.utils import MockDataset, translations_list_to_dataset


class TestFactory(TestCase):
    def test_collect(self):
        dataset_name, ext = 'tatoeba', 'bin'
        datasets = collect(dataset_name, save=True, ext=ext)
        assert_that(len(datasets), equal_to(1))
        dataset_path = get_dataset_path(dataset_name, ext=ext)
        assert_that(path.exists(dataset_path), equal_to(True))

    def test_wikimatrix(self):
        pass


class TestUtils(TestCase):
    def test_translations_list(self):
        targets, translations = MockDataset.targets(), MockDataset.translations()
        train_dataset, eval_dataset = translations_list_to_dataset(targets, translations, device='cpu')

        assert_that(len(train_dataset), equal_to(3))
        assert_that(len(eval_dataset), equal_to(1))
