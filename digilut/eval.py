"""WIP to prepare the evaluation functions."""

import pandas as pd
import torch
from torchmetrics.functional.detection import generalized_intersection_over_union


def compute_giou_scores_for_all_slides(
    preds_df: pd.DataFrame,
    target_df: pd.DataFrame,
    giou_threshold=0.50,
    beta: float = 2,
) -> pd.DataFrame:
    # Initialize a list to store results
    results = []

    # Get unique slide names from target_df (or preds_df, they should be the same)
    slide_names = target_df["slideName"].unique()

    for slide_name in slide_names:
        # Filter data frames for the given slide
        preds = preds_df[preds_df["slideName"] == slide_name]
        targets = target_df[target_df["slideName"] == slide_name]

        # Convert to tensors
        preds_tensor = torch.tensor(
            preds[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32
        )
        targets_tensor = torch.tensor(
            targets[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32
        )

        # Compute GIoU scores, return a (nb_preds x nb gt boxes) 2D matrix
        giou = generalized_intersection_over_union(
            preds_tensor, targets_tensor, aggregate=False
        )
        print(giou)

        # Threshold (1 if any of the giou scores for a box is above the threshold, 0 otherwise)
        giou_thresholded = giou >= giou_threshold
        giou_scores_per_gt_bbox = giou_thresholded.any(dim=0).int()
        giou_scores_per_pred_bbox = giou_thresholded.any(dim=1).int()

        tp = giou_scores_per_pred_bbox.sum().item()
        fp = len(giou_scores_per_pred_bbox) - tp
        nb_gt_bboxes_found = giou_scores_per_gt_bbox.sum().item()
        fn = len(giou_scores_per_gt_bbox) - nb_gt_bboxes_found

        # Compute metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f2score = (1 + beta**2) * tp / ((1 + beta**2) * tp + fp + (beta**2) * fn)

        result = {
            "slideName": slide_name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f2_score": f2score,
        }
        results.append(result)

    # Create a DataFrame from results
    return pd.DataFrame(results)


def compute_mean_f2_score(f2_scores_df) -> float:
    # Compute the mean F2 score
    mean_f2 = f2_scores_df["f2_score"].mean()
    return mean_f2


if __name__ == "__main__":
    # Sample data frames with predictions and ground truths for multiple slides
    preds_df = pd.DataFrame(
        {
            "slideName": ["slide1", "slide1", "slide1", "slide2", "slide2", "slide3"],
            "x1": [10, 20, 30, 40, 50, 70],
            "y1": [10, 20, 30, 40, 50, 70],
            "x2": [20, 30, 40, 50, 60, 80],
            "y2": [20, 30, 40, 50, 60, 80],
        }
    )

    target_df = pd.DataFrame(
        {
            "slideName": ["slide1", "slide1", "slide1", "slide2", "slide2", "slide3"],
            "x1": [10, 20, 30, 40, 50, 70],
            "y1": [10, 20, 30, 40, 50, 70],
            "x2": [20, 30, 40, 50, 60, 90],
            "y2": [20, 30, 40, 50, 60, 90],
        }
    )

    # Example usage
    output_df = compute_giou_scores_for_all_slides(preds_df, target_df)
    print(output_df)

    mean_f2_score = compute_mean_f2_score(output_df)
    print("mean f2 score:", mean_f2_score)
