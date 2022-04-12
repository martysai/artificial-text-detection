from abc import abstractmethod
from typing import List

import pandas as pd
from datasets import load_metric
from sacrebleu import BLEU
from tqdm import tqdm


class Metrics:
    """
    A simple class to calculate metrics.
    """
    def __init__(
        self,
        metrics_name: str,
        metrics_func: callable = None,
        pred_colname: str = "translations",
        trg_colname: str = "target"
    ) -> None:
        self.metrics_name = metrics_name
        self.metrics_func = metrics_func
        self.pred_colname = pred_colname
        self.trg_colname = trg_colname

    def compute(self, dataset: pd.DataFrame) -> List[float]:
        return [
            self.metrics_func(row, self.pred_colname, self.trg_colname)
            for _, row in tqdm(dataset.iterrows(), total=len(dataset))
        ]

    @abstractmethod
    def calc_metrics(self, sample: pd.Series) -> float:
        raise NotImplementedError


class SimilarityMetrics:
    """
    Define similarity metrics.
    """


class BLEUMetrics(Metrics):
    def __init__(self, metrics_name: str = "BLEU", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, **kwargs)
        self.bleu_metrics = BLEU()
        # load_metric(metrics_path or "")

    def calc_metrics(self, sample: pd.Series) -> float:
        bleu_result = self.bleu_metrics.sentence_score(sample[self.pred_colname], [sample[self.trg_colname]])
        return bleu_result.score


class METEORMetrics(Metrics):
    def __init__(self, metrics_name: str = "METEOR", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, **kwargs)
        self.meteor_metrics = load_metric("meteor")

    def calc_metrics(self, sample: pd.Series) -> float:
        meteor_result = self.meteor_metrics.compute(
            predictions=[sample[self.pred_colname]],
            references=[sample[self.trg_colname]]
        )
        return meteor_result["meteor"]


class TERMetrics(Metrics):
    def __init__(self, metrics_name: str = "TER", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, **kwargs)
        self.ter_metrics = load_metric("ter")

    def calc_metrics(self, sample: pd.Series) -> float:
        ter_result = self.ter_metrics.compute(
            predictions=[sample[self.pred_colname]],
            references=[[sample[self.trg_colname]]],
            case_sensitive=True
        )
        return ter_result["ter"]


class BLEURTMetrics(Metrics):
    def __init__(self, metrics_name: str = "BLEURT", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, **kwargs)
        self.bleurt_metrics = load_metric("bleurt", "BLEURT-20")

    def calc_metrics(self, sample: pd.Series) -> float:
        bleurt_result = self.bleurt_metrics.compute(
            predictions=[sample[self.pred_colname]],
            references=[[sample[self.trg_colname]]],
        )
        return bleurt_result["bleurt"]

# TODO: Comet, Cosine


METRICS_MAPPING = {
    "BLEU": BLEUMetrics,
    "METEOR": METEORMetrics,
    "TER": TERMetrics,
    "BLEURT": BLEURTMetrics,
    "Comet": CometMetrics,
    "Cosine": CosineMetrics
}


class Calculator:
    def __init__(self, path: str):
        self.dataset = pd.read_csv(path, sep="\t")

    def compute(self, metrics_names: List[str]) -> pd.DataFrame:
        for metrics_name in metrics_names:
            computed_values = METRICS_MAPPING[metrics_name].compute(self.dataset)
            self.dataset[metrics_name] = computed_values
        return self.dataset


def main() -> None:
    # TODO: define these variables
    metrics_names = []
    path = ...
    output_path = ...

    calculator = Calculator(path)

    metrics_df = calculator.compute(metrics_names)
    metrics_df.to_csv(output_path, index=False, sep="\t")


if __name__ == "__main__":
    main()
