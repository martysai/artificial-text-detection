import argparse
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import comet
import pandas as pd
from comet.models import CometModel
from datasets import load_metric
from lexicalrichness import LexicalRichness
from sacrebleu import BLEU
from tqdm import tqdm

from artificial_detection.arguments import form_proxy_args


class Metrics:
    """
    A simple class to calculate metrics.
    """
    def __init__(
        self,
        metrics_name: str,
        metrics_func: callable = None,
        pred_colname: str = "translations",
        trg_colname: str = "targets"
    ) -> None:
        self.metrics_name = metrics_name
        self.metrics_func = metrics_func
        self.pred_colname = pred_colname
        self.trg_colname = trg_colname

    def compute(self, dataset: pd.DataFrame) -> List[float]:
        return [
            self.metrics_func(row)
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
        return ter_result["score"]


class BLEURTMetrics(Metrics):
    def __init__(self, metrics_name: str = "BLEURT", model_path: str = None, **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, **kwargs)
        if model_path:
            self.bleurt_metrics = load_metric(model_path)
        else:
            self.bleurt_metrics = load_metric("bleurt", "BLEURT-20")

    def calc_metrics(self, sample: pd.Series) -> float:
        bleurt_result = self.bleurt_metrics.compute(
            predictions=[sample[self.pred_colname]],
            references=[[sample[self.trg_colname]]],
        )
        return bleurt_result["scores"]


class CometMetrics(Metrics):
    def __init__(self, metrics_name: str = "Comet", model_path: Optional[str] = None, device: str = None, **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, **kwargs)
        comet_metrics, _ = self.load_offline(model_path)
        self.device = device
        self.comet_metrics = comet_metrics

    @staticmethod
    def _prepare_data(sample: pd.Series) -> List[Dict[str, str]]:
        return [{
            "src": sample["sources"],
            "mt": sample["translations"],
            "ref": sample["targets"]
        }]

    def calc_metrics(self, sample: pd.Series) -> float:
        prepared_data = self._prepare_data(sample)
        seg_scores, sys_score = self.comet_metrics.predict(
            prepared_data, gpus=int(str(self.device).startswith("cuda"))
        )
        return seg_scores[0]

    @staticmethod
    def load_offline(model_path: Optional[str] = None) -> Tuple[CometModel, str]:
        if not model_path:
            model_path = comet.download_model("wmt20-comet-da")
        comet_model = comet.load_from_checkpoint(model_path)
        return comet_model, model_path


class BERTScoreMetrics(Metrics):
    def __init__(
        self,
        metrics_name: str = "BERTScore",
        model_path: str = None,
        baseline_path: str = None,
        device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, **kwargs)
        if model_path:
            self.bert_score_metrics = load_metric(model_path)
        else:
            self.bert_score_metrics = load_metric("bertscore")
        self.baseline_path = baseline_path
        self.device = device

    def calc_metrics(self, sample: pd.Series) -> float:
        bert_score_result = self.bert_score_metrics.compute(
            predictions=[sample[self.pred_colname]],
            references=[sample[self.trg_colname]],
            lang="ru",
            device=self.device,
            baseline_path=self.baseline_path
        )
        return bert_score_result["f1"][0]


class LexicalRichnessMetrics(Metrics):
    def __init__(self, metrics_name: str = "LexicalRichness", attr: str = "general", **kwargs) -> None:
        super().__init__(metrics_name, **kwargs)
        self.richness = LexicalRichness
        self.attr = attr

    def calc_metrics(self, sample: pd.Series) -> float:
        richness_result = self.richness(sample["translations"])
        try:
            return getattr(richness_result, self.attr)
        except ZeroDivisionError:
            return 0.0


class LexicalRichnessWords(LexicalRichnessMetrics):
    def __init__(self, metrics_name: str = "LexicalRichnessWords", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, attr="words", **kwargs)


class LexicalRichnessTerms(LexicalRichnessMetrics):
    def __init__(self, metrics_name: str = "LexicalRichnessTerms", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, attr="terms", **kwargs)


class LexicalRichnessTTR(LexicalRichnessMetrics):
    def __init__(self, metrics_name: str = "LexicalRichnessTTR", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, attr="ttr", **kwargs)


class LexicalRichnessRTTR(LexicalRichnessMetrics):
    def __init__(self, metrics_name: str = "LexicalRichnessRTTR", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, attr="rttr", **kwargs)


class LexicalRichnessCTTR(LexicalRichnessMetrics):
    def __init__(self, metrics_name: str = "LexicalRichnessCTTR", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, attr="cttr", **kwargs)


class LexicalRichnessMTLD(LexicalRichnessMetrics):
    def __init__(self, metrics_name: str = "LexicalRichnessMTLD", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, attr="mtld", **kwargs)


class LexicalRichnessHerdan(LexicalRichnessMetrics):
    def __init__(self, metrics_name: str = "LexicalRichnessHerdan", **kwargs) -> None:
        super().__init__(metrics_name, metrics_func=self.calc_metrics, attr="Herdan", **kwargs)


METRICS_MAPPING = {
    "BLEU": BLEUMetrics,
    "METEOR": METEORMetrics,
    "TER": TERMetrics,
    "BLEURT": BLEURTMetrics,
    "Comet": CometMetrics,
    "BERTScore": BERTScoreMetrics,
    "LexicalRichnessWords": LexicalRichnessWords,
    "LexicalRichnessTerms": LexicalRichnessTerms,
    "LexicalRichnessTTR": LexicalRichnessTTR,
    "LexicalRichnessRTTR": LexicalRichnessRTTR,
    "LexicalRichnessCTTR": LexicalRichnessCTTR,
    "LexicalRichnessMTLD": LexicalRichnessMTLD,
    "LexicalRichnessHerdan": LexicalRichnessHerdan,
}


class Calculator:
    def __init__(
        self,
        df_or_path: Union[pd.DataFrame, str],
        model_specific_dict: Dict[str, Any]
    ):
        if isinstance(df_or_path, str):
            self.dataset = pd.read_csv(df_or_path, sep="\t")
        else:
            self.dataset = df_or_path
        self.model_specific_dict = model_specific_dict

    def instantiate_metrics(self, metrics_names: List[str]) -> Dict[str, Metrics]:
        return {
            metric_name: METRICS_MAPPING[metric_name](metrics_name=metric_name, **self.model_specific_dict[metric_name])
            for metric_name in metrics_names
        }

    def compute(self, metrics_names: List[str]) -> pd.DataFrame:
        metrics_instances = self.instantiate_metrics(metrics_names)

        for metrics_name in metrics_names:
            computed_values = metrics_instances[metrics_name].compute(self.dataset)
            self.dataset[metrics_name] = computed_values
        return self.dataset


def form_model_specific_dict(args: argparse.ArgumentParser) -> Dict[str, Any]:
    model_specific_dict = {
        "Comet": {
            "device": args.device,
            "model_path": args.model_path,
        },
        "BERTScore": {
            "baseline_path": args.baseline_path,
            "device": args.device,
            "model_path": args.model_path,
        },
    }
    return defaultdict(dict, model_specific_dict)


def main() -> None:
    args = form_proxy_args()
    model_specific_dict = form_model_specific_dict(args)

    calculator = Calculator(df_or_path=args.df_path, model_specific_dict=model_specific_dict)

    metrics_df = calculator.compute(args.metrics_names)
    metrics_df.to_csv(args.output_path, index=False, sep="\t")


if __name__ == "__main__":
    main()
