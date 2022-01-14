import argparse
from typing import Any, Dict, Optional

import pandas as pd
import tqdm
import numpy as np

from detection.arguments import form_args
from detection.data.generate_language_model import (
    check_input_paragraph,
    check_output_paragraph,
    generate_language_model,
    retrieve_prefix,
    super_maximal_repeat,
    trim_output_paragraph,
    parse_collection_on_repeats,
)
from detection.models.const import (
    SEMI_SUPERVISED_HUMAN_RATE,
    SMR_REPEAT_RATE,
    LM_LENGTH_LOWER_BOUND,
    SMR_LENGTH_LOWER_BOUND,
)
from detection.models.detectors import SimpleDetector


class UnsupervisedBaseline:
    """
    TODO
    """

    def __init__(
        self,
        args: argparse.Namespace,
        labeled_df: pd.DataFrame = None,
        mode: str = "unsupervised",
        use_wandb: bool = False,
        sample: str = "topk",
    ):
        if sample not in ["topk", "nucl"]:
            raise ValueError("Wrong value for sample")
        self.detector = SimpleDetector(args=args, use_wandb=use_wandb)
        self.labeled_df = labeled_df
        if mode not in ["semi-supervised", "unsupervised"]:
            raise ValueError("Possible mode options are 'semi-supervised' and 'unsupervised'")
        self.mode = mode
        self.sample = sample  # TODO: расширить sample: top_k = 10, p = 0.8

    @staticmethod
    def process(df: pd.DataFrame, lm_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with human paragraphs.
        lm_params: Optional[Dict[str, Any]], default=None
            Additional language model parameters.

        Returns
        -------
        pd.DataFrame
            Processed with GPT-2 dataset consisting of human/machine - continued paragraphs.
        """
        labeled_data = []
        paragraphs = df.values.reshape(-1,).tolist()
        paragraphs = list(filter(check_input_paragraph, paragraphs))
        generated_paragraphs = generate_language_model(paragraphs, lm_params=lm_params)
        for i, generated_paragraph in tqdm.tqdm(enumerate(generated_paragraphs)):
            if check_output_paragraph(generated_paragraph):
                labeled_data.extend(
                    [
                        {"text": trim_output_paragraph(generated_paragraph), "target": "machine"},
                        {"text": trim_output_paragraph(paragraphs[i]), "target": "human"},
                    ]
                )
        labeled_df = pd.DataFrame(data=labeled_data, columns=["text", "target"])
        return labeled_df

    @staticmethod
    def label_with_repeats(
        df: pd.DataFrame,
        target_name: str = "target",
        save_repeats: bool = False,
        collection_length: int = LM_LENGTH_LOWER_BOUND,
        smr_length: int = SMR_LENGTH_LOWER_BOUND,
        positive_repeat_rate: float = SMR_REPEAT_RATE,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Label the dataset based on the algorithm provided in https://arxiv.org/abs/2111.02878.
        We use suffix trees for super-maximal repeats counting.

        Parameters
        ----------
        df: pd.DataFrame
            TODO-Docs
        target_name: str
        save_repeats: bool
        collection_length: int
        smr_length: int
        positive_repeat_rate: float
        seed: int
            Random seed.

        Returns
        -------
        pd.DataFrame
            Labeled dataframe for the further unsupervised baseline training process.
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
        df[target_name] = np.array(smr_labels)
        return df

    @staticmethod
    def semi_supervise(
        df: pd.DataFrame,
        target_name: str = "target",
        supervise_rate: float = SEMI_SUPERVISED_HUMAN_RATE,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        TODO-Docs
        """
        supervised_sample = df.head(int(supervise_rate * len(df)))
        supervised_sample = supervised_sample[supervised_sample[target_name] == "human"]
        unsupervised_sample = df.tail(int((1.0 - supervise_rate) * len(df)))
        unsupervised_sample = UnsupervisedBaseline.label_with_repeats(
            unsupervised_sample, target_name=target_name, save_repeats=True, seed=seed
        )
        return supervised_sample.append(unsupervised_sample)

    def fit(self, df: pd.DataFrame, target_name: str = "target", force: bool = False):
        """
        TODO-Docs
        """
        # TODO: обобщить до двух опций: unsupervised и semi_supervised
        if (not self.labeled_df) or force:
            print("here in process and semi supervise")
            # Here is target_name is set to default in order to distinguish labeled target and the ground truth one.
            df = self.process(df)
            self.labeled_df = UnsupervisedBaseline.semi_supervise(df, target_name=target_name)
        X, y = self.labeled_df["text"], self.labeled_df[target_name]
        print("fitting the detector")
        self.detector.fit(X, y)

    def predict(self, X_test: pd.DataFrame):
        return self.detector.predict(X_test)


if __name__ == "__main__":
    main_args = form_args()

    supervised_df = pd.read_csv(main_args.detector_dataset_path)
    baseline = UnsupervisedBaseline(args=main_args, use_wandb=True, labeled_df=supervised_df)
    baseline.fit(supervised_df, target_name=main_args.target_name)
    y_pred = baseline.predict(supervised_df)
    print(y_pred.shape)
