import pandas as pd
import pytest
from hamcrest import assert_that, close_to

from artificial_detection.data.proxy import BLEUMetrics, CometMetrics, METEORMetrics, TERMetrics


@pytest.fixture
def mock_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sources": "Il va à la bibliothèque pour lire des livres.",
                "translations": "Он едет в библиотеку, чтобы читать книги.",
                "targets": "Он ходит в библиотеку, чтобы читать книги.",
            },
            {
                "sources": "It is no use crying over spilt milk.",
                "translations": "Нет смысла плакать над пролитым молоком.",
                "targets": "Слезами горю не поможешь.",
            },
        ]
    )


def test_bleu(mock_dataset: pd.DataFrame) -> None:
    metrics = BLEUMetrics()
    scores = metrics.compute(mock_dataset)
    gt_scores = [75.062, 6.567]
    for i, score in enumerate(scores):
        assert_that(score, close_to(gt_scores[i], 0.01))


def test_meteor(mock_dataset: pd.DataFrame) -> None:
    metrics = METEORMetrics()
    scores = metrics.compute(mock_dataset)
    gt_scores = [0.882, 0.096]
    for i, score in enumerate(scores):
        assert_that(score, close_to(gt_scores[i], 0.01))


def test_ter(mock_dataset: pd.DataFrame) -> None:
    metrics = TERMetrics()
    scores = metrics.compute(mock_dataset)
    gt_scores = [14.286, 150.0]
    for i, score in enumerate(scores):
        assert_that(score, close_to(gt_scores[i], 0.01))


def test_bleurt(mock_dataset: pd.DataFrame) -> None:
    pass


def test_comet(mock_dataset: pd.DataFrame) -> None:
    metrics = CometMetrics()
    scores = metrics.compute(mock_dataset)
    gt_scores = [0.9, 0.9]
    for i, score in enumerate(scores):
        assert_that(score, close_to(gt_scores[i], 0.01))


def test_cosine(mock_dataset: pd.DataFrame) -> None:
    pass
