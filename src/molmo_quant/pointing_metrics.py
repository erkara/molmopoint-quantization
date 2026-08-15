"""PixMo pointing metrics compatible with AllenAI's Molmo evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def is_point_in_region(point: tuple[float, float] | np.ndarray, mask: np.ndarray) -> bool:
    """Use AllenAI's nearest-pixel convention and bounds behavior."""
    height, width = mask.shape
    x, y = point
    x_int = int(round(float(x)))
    y_int = int(round(float(y)))
    if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
        return False
    return bool(mask[y_int, x_int])


def score_pointing_prediction(
    predicted_points: list[dict[str, Any]] | list[tuple[float, float]],
    gt_points: np.ndarray,
    masks: list[np.ndarray] | np.ndarray,
) -> dict[str, float]:
    """Compute the official macro precision/recall/F1 inputs for one example."""
    predictions = np.asarray(
        [
            (float(point["x"]), float(point["y"]))
            if isinstance(point, dict)
            else (float(point[0]), float(point[1]))
            for point in predicted_points
        ],
        dtype=float,
    ).reshape(-1, 2)
    ground_truth = np.asarray(gt_points, dtype=float).reshape(-1, 2)
    mask_arrays = [np.asarray(mask, dtype=bool) for mask in masks]

    if len(ground_truth) == 0:
        value = float(len(predictions) == 0)
        return {"precision": value, "recall": value, "f1": value}
    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    row_ind, col_ind = linear_sum_assignment(cdist(predictions, ground_truth))
    correct = sum(
        is_point_in_region(predictions[pred_index], mask_arrays[gt_index])
        for pred_index, gt_index in zip(row_ind, col_ind, strict=True)
    )
    precision = correct / len(predictions)
    recall = correct / len(mask_arrays)
    f1 = (
        0.0
        if precision == 0 or recall == 0
        else 2 * precision * recall / (precision + recall + 1e-10)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def absolute_gt_points(points: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    """Convert the dataset's 0-100 point coordinates to original image pixels."""
    return np.asarray(
        [[float(point["x"]) * width / 100, float(point["y"]) * height / 100] for point in points],
        dtype=float,
    ).reshape(-1, 2)


def matched_prediction_distances(
    left_points: list[dict[str, Any]], right_points: list[dict[str, Any]]
) -> list[float]:
    """Return one-to-one pixel distances for agreement with the BF16 output."""
    if not left_points or not right_points:
        return []
    left = np.asarray([[point["x"], point["y"]] for point in left_points], dtype=float)
    right = np.asarray([[point["x"], point["y"]] for point in right_points], dtype=float)
    distances = cdist(left, right)
    row_ind, col_ind = linear_sum_assignment(distances)
    return [float(distances[i, j]) for i, j in zip(row_ind, col_ind, strict=True)]
