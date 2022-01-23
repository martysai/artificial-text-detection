import os.path as path
from unittest import TestCase

from hamcrest import assert_that, equal_to

from detection.data.factory import collect, load_wikimatrix
from detection.utils import MockDataset, get_dataset_path, translations_to_torch_dataset
from tests import skip_github


class TestFactory(TestCase):
    def test_collect_tatoeba(self):
        dataset_name, ext = "tatoeba", "bin"
        datasets = collect(dataset_name, save=True, ext=ext)
        assert_that(len(datasets), equal_to(4))
        dataset_path = get_dataset_path(dataset_name, langs=["en", "ru"], ext=ext)
        assert_that(path.exists(dataset_path), equal_to(True))
        dataset_path = get_dataset_path(dataset_name, langs=["fr", "ru"], ext=ext)
        assert_that(path.exists(dataset_path), equal_to(True))

    @skip_github
    def test_collect_wikimatrix(self):
        dataset = load_wikimatrix(lang1="en", lang2="ru")
        assert_that(len(dataset), equal_to(1661908))
        assert_that(dataset[0]["ru"], equal_to("Какую же из милостей вашего Господа вы считаете ложью?»."))
        assert_that(dataset[0]["en"], equal_to('The glory of the Lord has risen upon thee".'))

    def test_languages(self):
        pass


class TestUtils(TestCase):
    def test_translations_list(self):
        targets, translations = MockDataset.targets(), MockDataset.translations()
        dataset = translations_to_torch_dataset(targets, translations, device="cpu")
        train_dataset, eval_dataset = dataset.split()

        assert_that(len(train_dataset), equal_to(3))
        assert_that(len(eval_dataset), equal_to(1))
