from typing import Any, Dict, Optional

import pandas as pd

from detection.data.generate_language_model import (
    check_input_paragraph, check_output_paragraph, generate_language_model, retrieve_prefix,
    super_maximal_repeat, trim_output_paragraph
)
from detection.models.const import SEMI_SUPERVISED_HUMAN_RATE
from detection.models.detectors import SimpleDetector


class UnsupervisedBaseline:
    """
    TODO
    """
    def __init__(self, sample: str = "topk"):
        if sample not in ["topk", "nucl"]:
            raise ValueError("Wrong value for sample")
        self.detector = SimpleDetector()
        self.sample = sample  # TODO: расширить sample: top_k = 10, p = 0.8
        self.unsupervised_df = None

    @staticmethod
    def process(df: pd.DataFrame, lm_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        df: pd.DataFrame
            TODO
        lm_params: Optional[Dict[str, Any]], default=None
            TODO

        Returns
        -------
            TODO
        """
        unsupervised_data = []
        paragraphs = df.values.reshape(-1,).tolist()
        paragraphs = list(filter(check_input_paragraph, paragraphs))
        generated_paragraphs = generate_language_model(paragraphs, lm_params=lm_params)
        for i, generated_paragraph in enumerate(generated_paragraphs):
            if check_output_paragraph(generated_paragraph):
                unsupervised_data.extend([
                    {
                        "text": trim_output_paragraph(generated_paragraph),
                        "target": "machine"
                    },
                    {
                        "text": paragraphs[i],
                        "target": "human"
                    }
                ])
        unsupervised_df = pd.DataFrame(data=unsupervised_data, columns=["text", "target"])
        return unsupervised_df

    def semi_supervise(self, df: pd.DataFrame) -> pd.DataFrame:
        supervised_sample = df.head(SEMI_SUPERVISED_HUMAN_RATE * len(df))
        supervised_sample = supervised_sample[supervised_sample["target"] == "human"]
        semi_supervised_df = df.tail((1.0 - SEMI_SUPERVISED_HUMAN_RATE) * len(df))
        semi_supervised_df["repeats"] = semi_supervised_df["text"].apply(super_maximal_repeat)
        # TODO: сэмплируем, выбираем таргеты
        # TODO: написать тест, что разметка проходит корректно
        return semi_supervised_df

    def fit(self, df: pd.DataFrame, force: bool = False):
        if (not self.unsupervised_df) or (self.unsupervised_df and force):
            df = self.process(df)
            self.unsupervised_df = self.semi_supervise(df)
        X, y = self.unsupervised_df["text"], self.unsupervised_df["target"]
        self.detector.fit(X, y)

    def predict(self, X_test: pd.DataFrame):
        return self.detector.predict(X_test)
