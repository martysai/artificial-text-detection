from unittest import TestCase

import pandas as pd
from hamcrest import assert_that, close_to, equal_to, instance_of
from transformers import BertForSequenceClassification, DistilBertTokenizerFast

from artificial_detection.arguments import form_args
from artificial_detection.models.const import HF_MODEL_NAME
from artificial_detection.models.detectors import SimpleDetector


class TestDetectors(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.args = form_args()
        cls.args.epochs = 1
        cls.simple_detector = SimpleDetector(args=cls.args, use_wandb=False)

        cls.X = pd.DataFrame([
            {"text": "добрый день. это какой-то сэмпл."},
            {"text": "здравствуйте. а это другой сэмпл."}
        ], columns=["text"])
        cls.y = pd.DataFrame([
            {"target": "human"},
            {"target": "machine"}
        ], columns=["target"])
        cls.X_test = pd.DataFrame([
            {"text": "третий сэмпл"}
        ], columns=["text"])

    def test_simple_detector_parameters(self) -> None:
        assert_that(self.simple_detector.training_args.device.type, equal_to("cpu"))
        assert_that(self.simple_detector.training_args.label_smoothing_factor, equal_to(0.))
        assert_that(self.simple_detector.training_args.run_name, equal_to("default"))
        assert_that(self.simple_detector.model, instance_of(BertForSequenceClassification))
        assert_that(self.simple_detector.model.config.vocab_size, equal_to(29564))

    def test_simple_detector_fit(self) -> None:
        self.simple_detector.fit(self.X, self.y)
        y_pred = self.simple_detector.predict(self.X_test)
        assert_that(y_pred["target"].values, equal_to(["human"]))

    def test_simple_detector_predict_proba(self) -> None:
        self.simple_detector.fit(self.X, self.y)
        y_proba = self.simple_detector.predict_proba(self.X_test)
        # TODO: зафиксировать предсказания
        # assert_that(y_proba["proba"].values[0], close_to(-0.1628, 0.01))

    def test_tokenizer(self) -> None:
        tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL_NAME)
        assert_that(tokenizer(["предложение"])['input_ids'][0], equal_to([2, 24889, 3]))
