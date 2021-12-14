from typing import Any, Dict, Optional

import pandas as pd

from detection.data.generate_LM import check_paragraph, generate_LM, retrieve_prefix
from detection.models.detectors import SimpleDetector


class UnsupervisedBaseline:
    """
    """
    def __init__(self, sample: str = "topk"):
        if sample not in ["topk", "nucl"]:
            raise ValueError("Wrong value for sample")
        self.detector = SimpleDetector()
        self.sample = sample

    def process(self, df: pd.DataFrame, lm_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        for i, row in df.iterrows():
            paragraph = row["text"]
            if not check_paragraph(paragraph):
                continue
            # TODO-LM: делать через батч
            generated_paragraph = generate_LM([paragraph], lm_params=lm_params)
        return result

    def fit(self, df: pd.DataFrame):
        df_targets = self.process(df)
        X, y = df_targets["text"], df_targets["target"]
        self.detector.fit(X, y)

    def predict(self, X_test: pd.DataFrame):
        return self.detector.predict(X_test)
