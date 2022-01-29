from unittest import TestCase

import pandas as pd
import pytest
from hamcrest import assert_that, equal_to

from artificial_detection.arguments import form_args
from artificial_detection.pipeline import pipeline
from artificial_detection.unsupervised_baseline import run_unsupervised_baseline_fit, transform_unsupervised_metrics
from artificial_detection.utils import fix_random_seed
from tests import skip_github


class TestPipeline(TestCase):
    @pytest.mark.timeout(30)
    @skip_github
    def test_run(self) -> None:
        args = form_args()
        args.run_name = "test_run"
        args.epochs = 1
        args.size = 8
        args.is_mock_data = True  # TODO: вспомнить как работают mock-данные
        trainers = pipeline(args)
        assert_that(len(trainers), equal_to(4))


class TestUnsupervisedPipeline(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.supervised_df = pd.DataFrame(
            [
                {
                    "text": "Министерство народного просвещения, в виду происходящих чрезвычайных событий, признало.",
                    "target": "machine",
                    "unsupervised_target": "human",
                },
                {
                    "text": "Министерство народного просвещения, в виду происходящих чрезвычайных событий, признало.",
                    "target": "human",
                    "unsupervised_target": "machine",
                },
                {
                    "text": "Штабс-капитан П. Е. Крылов в письме Н. Э. Баскакову "
                    "31 января 1916 года сообщает, что в штабе.",
                    "target": "machine",
                    "unsupervised_target": "human",
                },
            ]
        )
        cls.test_df = pd.DataFrame(
            [
                {
                    "text": "Штабс-капитан П. Н. Нестеров на днях, увидев в районе Желтиева, в Галиции.",
                    "target": "human",
                    "unsupervised_target": "human",
                },
                {
                    "text": "Фотограф-корреспондент Daily Mirror рассказывает случай, который порадует всех друзей.",
                    "target": "machine",
                    "unsupervised_target": "machine",
                },
            ]
        )
        cls.seed = 1206

    def setUp(self) -> None:
        fix_random_seed(self.seed)

    @pytest.mark.timeout(30)
    @skip_github
    def test_unsupervised_run(self) -> None:
        main_args = form_args()
        main_args.train_batch = 1
        main_args.eval_batch = 1
        main_args.epochs = 3
        main_args.target_name = "target"

        baseline = run_unsupervised_baseline_fit(main_args, self.supervised_df)

        y_pred = baseline.predict(pd.DataFrame(self.test_df["text"], columns=["text"]))
        metrics = transform_unsupervised_metrics(self.test_df, y_pred, main_args.target_name)
        assert_that(len(list(metrics.keys())), equal_to(5))
        assert_that(metrics["accuracy_score"], equal_to(0.5))
        assert_that(metrics["f1_score"], equal_to(0.0))
