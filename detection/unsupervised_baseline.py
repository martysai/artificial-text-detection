import argparse
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from detection.arguments import form_args
from detection.data.generate_language_model import (
    check_input_paragraph, check_output_paragraph, generate_language_model, retrieve_prefix,
    super_maximal_repeat, trim_output_paragraph, parse_collection_on_repeats
)
from detection.models.const import (
    SEMI_SUPERVISED_HUMAN_RATE, SMR_REPEAT_RATE,
    LM_LENGTH_LOWER_BOUND, SMR_LENGTH_LOWER_BOUND
)
from detection.models.detectors import SimpleDetector


class UnsupervisedBaseline:
    """
    TODO
    """
    def __init__(self, args: argparse.Namespace, sample: str = "topk"):
        if sample not in ["topk", "nucl"]:
            raise ValueError("Wrong value for sample")
        self.detector = SimpleDetector(args=args)
        self.sample = sample  # TODO: расширить sample: top_k = 10, p = 0.8
        self.semi_supervised_df = None

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

    @staticmethod
    def label_with_repeats(
        df: pd.DataFrame,
        save_repeats: bool = False,
        collection_length: int = LM_LENGTH_LOWER_BOUND,
        smr_length: int = SMR_LENGTH_LOWER_BOUND,
        positive_repeat_rate: float = SMR_REPEAT_RATE,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        TODO-Docs
        """
        parapgraphs_list = df["text"].values.tolist()
        if save_repeats:
            # TODO: save repeats
            pass
        repeats = parse_collection_on_repeats(
            parapgraphs_list, collection_length=collection_length, smr_length=smr_length
        )
        positive_repeats = np.random.RandomState(seed).choice(repeats, int(positive_repeat_rate * len(repeats)))
        smr_labels = []
        for i, row in df.iterrows():
            paragraph = row["text"]
            is_positive = any([repeat in paragraph for repeat in positive_repeats])
            smr_labels.append("machine" if is_positive else "human")
        df["target"] = np.array(smr_labels)
        return df

    @staticmethod
    def semi_supervise(
        df: pd.DataFrame,
        supervise_rate: float = SEMI_SUPERVISED_HUMAN_RATE,
        seed: int = 42
    ) -> pd.DataFrame:
        supervised_sample = df.head(int(supervise_rate * len(df)))
        supervised_sample = supervised_sample[supervised_sample["target"] == "human"]
        unsupervised_sample = df.tail(int((1.0 - supervise_rate) * len(df)))
        unsupervised_sample = UnsupervisedBaseline.label_with_repeats(unsupervised_sample, save_repeats=True, seed=seed)
        return supervised_sample.append(unsupervised_sample)

    def fit(self, df: pd.DataFrame, force: bool = False):
        # TODO: обобщить до двух опций: unsupervised и semi_supervised
        if (not self.semi_supervised_df) or (self.semi_supervised_df and force):
            df = self.process(df)
            self.semi_supervised_df = UnsupervisedBaseline.semi_supervise(df)
        X, y = self.semi_supervised_df["text"], self.semi_supervised_df["target"]
        self.detector.fit(X, y)

    def predict(self, X_test: pd.DataFrame):
        return self.detector.predict(X_test)


if __name__ == "__main__":
    # TODO: дописать запуск на настоящих данных
    dir_path = path.dirname(path.dirname(path.realpath(__file__)))
    dvc_path = path.join(dir_path, "resources/data")
    news_df = pd.read_csv(path.join(dvc_path, "lenta/lenta_sample.csv"), nrows=20)

    args = form_args()
    baseline = UnsupervisedBaseline(args=args)
    baseline.fit(news_df)
    pass
