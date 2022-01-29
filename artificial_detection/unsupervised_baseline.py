import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from artificial_detection.arguments import form_args
from artificial_detection.data.generate_language_model import (
    check_input_paragraph,
    check_output_paragraph,
    generate_language_model,
    parse_collection_on_repeats,
    trim_output_paragraph,
)
from artificial_detection.models.const import (
    LM_LENGTH_LOWER_BOUND,
    METRIC_SKLEARN_NAMES,
    SEMI_SUPERVISED_HUMAN_RATE,
    SMR_LENGTH_LOWER_BOUND,
    SMR_REPEAT_RATE,
)
from artificial_detection.models.detectors import SimpleDetector


class UnsupervisedBaseline:
    """
    The baseline based on suffix trees which is motivated by the unsupervised training of the detector.
    Paper: https://arxiv.org/abs/2111.02878

    Attributes
    ----------
    detector: SimpleDetector
    labeled_df: pd.DataFrame

    mode: str
        Set the training option. Possible values: ["semi-supervised", "unsupervised"].
    sample: str
        Decoding strategy. Currently supported options: ["topk", "nucleus"].
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
    def process(
        df: pd.DataFrame,
        lm_params: Optional[Dict[str, Any]] = None,
        is_sentence: bool = True,
        cut_num: int = 1,
        input_lower_bound: int = None,
        output_lower_bound: int = None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with human paragraphs.
        lm_params: Optional[Dict[str, Any]], default=None
            Additional language model parameters.
        is_sentence: bool
            Flag defining that we split paragraphs by sentences or by tokens. Default is True.
        cut_num: int
            Number of sentences/tokens depending on is_sentence for paragraphs prefixes retrieval.
        input_lower_bound: int
            Possible input length for a given prompt.
        output_lower_bound: int
            Possible output length for a generated text.

        Returns
        -------
        pd.DataFrame
            Processed with GPT-2 dataset consisting of human/machine - continued paragraphs.
        """
        labeled_data = []
        paragraphs = df.values.reshape(-1,).tolist()
        paragraphs = list(filter(lambda p: check_input_paragraph(p, lower_bound=input_lower_bound), paragraphs))
        generated_paragraphs = generate_language_model(
            paragraphs, is_sentence=is_sentence, cut_num=cut_num, lm_params=lm_params
        )
        for i, generated_paragraph in tqdm.tqdm(enumerate(generated_paragraphs)):
            if check_output_paragraph(generated_paragraph, lower_bound=output_lower_bound):
                labeled_data.extend(
                    [
                        {"text": trim_output_paragraph(generated_paragraph), "target": "machine"},
                        {"text": trim_output_paragraph(paragraphs[i]), "target": "human"},
                    ]
                )
        labeled_df = pd.DataFrame(data=labeled_data, columns=["text", "target"])
        return labeled_df

    @staticmethod
    def label_by_given_repeats() -> List[str]:
        # TODO: add the entrypoint for labeling
        pass

    @staticmethod
    def label_with_repeats(
        df: pd.DataFrame,
        target_name: str = "target",
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
            The training dataset.
        target_name: str
            Name of column in the training dataset to be fed. Default is "target".
        collection_length: int
            TODO
        smr_length: int
        positive_repeat_rate: float
        seed: int
            Random seed.

        Returns
        -------
        pd.DataFrame
            Labeled dataframe for the further unsupervised baseline training process.
        """
        paragraphs_list = df["text"].values.tolist()
        repeats = parse_collection_on_repeats(
            paragraphs_list, collection_length=collection_length, smr_length=smr_length
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
            unsupervised_sample, target_name=target_name, seed=seed
        )
        return supervised_sample.append(unsupervised_sample)

    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        target_name: str = "target",
        force: bool = False,
        is_sentence: bool = True,
        cut_num: int = 1
    ) -> None:
        """
        Fit the baseline.

        Parameters
        ----------
        df: pd.DataFrame
            The training dataset. Overrides self.labeled_df if force=True.
        target_name: str
            Name of column in the training dataset to be fed. Default is "target".
        force: bool
            Replace self.labeled_df. Default is False which means that do not preprocess.
        is_sentence: bool
            Flag defining that we split paragraphs by sentences or by tokens. Default is True.
        cut_num: int
            Number of sentences/tokens depending on is_sentence for paragraphs prefixes retrieval.
        """
        # TODO: обобщить до двух опций: unsupervised и semi_supervised
        if isinstance(df, pd.DataFrame) and (not isinstance(self.labeled_df, pd.DataFrame)) or force:
            # Here is target_name is set to default in order to distinguish labeled target and the ground truth one.
            df = self.process(df, is_sentence=is_sentence, cut_num=cut_num)
            self.labeled_df = UnsupervisedBaseline.semi_supervise(df, target_name=target_name)
        X, y = self.labeled_df["text"].to_frame(), self.labeled_df[target_name].to_frame()
        self.detector.fit(X, y)

    def predict(self, X_test: pd.DataFrame, device: Optional[str] = "cpu"):
        return self.detector.predict(X_test, device=device)

    def predict_proba(self, X_test: pd.DataFrame):
        return self.detector.predict_proba(X_test)


def run_unsupervised_baseline_fit(args: argparse.Namespace, df: pd.DataFrame) -> UnsupervisedBaseline:
    """
    For given settings and a dataset fit the unsupervised baseline.
    """
    baseline = UnsupervisedBaseline(args=args, use_wandb=True, labeled_df=df)
    baseline.fit(df, target_name=args.unsupervised_target_name)
    return baseline


def transform_unsupervised_metrics(test_df: pd.DataFrame, y_pred: pd.DataFrame, target_name: str) -> Dict[str, float]:
    """
    Transform test predictions with LabelEncoder.
    """
    lec = LabelEncoder()
    y_pred = lec.fit_transform(y_pred.values.ravel())
    y_true = lec.fit_transform(test_df[target_name].values.ravel())
    test_metrics = {"pearson": pearsonr(y_true, y_pred)}
    for metric in METRIC_SKLEARN_NAMES:
        test_metrics.update({metric: getattr(metrics, metric)(y_true, y_pred)})
    return test_metrics


def main():
    main_args = form_args()
    supervised_df = pd.read_csv(main_args.detector_dataset_path, sep="\t", lineterminator="\n")
    baseline = run_unsupervised_baseline_fit(main_args, supervised_df)
    test_df = pd.read_csv(main_args.detector_dataset_test_path, sep="\t", lineterminator="\n")
    y_pred = baseline.predict(pd.DataFrame(test_df["text"], columns=["text"]), device="cpu")
    metrics_dict = transform_unsupervised_metrics(test_df, y_pred, main_args.target_name)
    print(metrics_dict)


if __name__ == "__main__":
    main()
