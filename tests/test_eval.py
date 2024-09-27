import pandas as pd
import pytest

from digilut.eval import compute_giou_scores_for_all_slides, compute_mean_f2_score


@pytest.fixture
def preds_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slideName": ["slide1", "slide1", "slide1", "slide2", "slide2", "slide3"],
            "x1": [10, 20, 30, 40, 50, 70],
            "y1": [10, 20, 30, 40, 50, 70],
            "x2": [20, 30, 40, 50, 60, 80],
            "y2": [20, 30, 40, 50, 60, 80],
        }
    )


@pytest.fixture
def target_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slideName": ["slide1", "slide1", "slide1", "slide2", "slide2", "slide3"],
            "x1": [10, 20, 30, 40, 50, 70],
            "y1": [10, 20, 30, 40, 50, 70],
            "x2": [20, 30, 40, 50, 60, 90],
            "y2": [20, 30, 40, 50, 60, 90],
        }
    )


def test_eval(preds_df, target_df):
    output = compute_giou_scores_for_all_slides(preds_df, target_df)
    expected = pd.DataFrame(
        {
            "slideName": {0: "slide1", 1: "slide2", 2: "slide3"},
            "tp": {0: 3, 1: 2, 2: 0},
            "fp": {0: 0, 1: 0, 2: 1},
            "fn": {0: 0, 1: 0, 2: 1},
            "precision": {0: 1.0, 1: 1.0, 2: 0.0},
            "recall": {0: 1.0, 1: 1.0, 2: 0.0},
            "f2_score": {0: 1.0, 1: 1.0, 2: 0.0},
        }
    )
    pd.testing.assert_frame_equal(output, expected)


def test_mean_f2_score(preds_df, target_df):
    output_df = compute_giou_scores_for_all_slides(preds_df, target_df)
    mean_f2_score = compute_mean_f2_score(output_df)
    assert mean_f2_score == 0.6666666666666666
