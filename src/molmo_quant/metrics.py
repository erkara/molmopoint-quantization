"""Small agreement metrics shared by smoke and dataset evaluation."""

from __future__ import annotations

import math
from itertools import permutations
from typing import Any


def compare_to_baseline(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_points = baseline.get("points", [])
    candidate_points = candidate.get("points", [])
    distances = matched_point_distances(baseline_points, candidate_points)
    return {
        "candidate_variant": candidate.get("variant"),
        "exact_token_agreement": baseline.get("generated_token_ids")
        == candidate.get("generated_token_ids"),
        "point_count_agreement": len(baseline_points) == len(candidate_points),
        "baseline_point_count": len(baseline_points),
        "candidate_point_count": len(candidate_points),
        "matched_point_distances_px": distances,
        "mean_matched_point_distance_px": (
            sum(distances) / len(distances) if distances else None
        ),
        "max_matched_point_distance_px": max(distances) if distances else None,
        "baseline_parse_success": bool(baseline.get("parse_success")),
        "candidate_parse_success": bool(candidate.get("parse_success")),
    }


def matched_point_distances(
    baseline_points: list[dict[str, Any]], candidate_points: list[dict[str, Any]]
) -> list[float]:
    """Return minimum-total-distance point matching for small smoke outputs.

    The exact permutation search is deliberate: smoke prompts normally emit a
    handful of points. Dataset-scale evaluation can replace this with a
    Hungarian solver if examples with large point sets make that necessary.
    """

    count = min(len(baseline_points), len(candidate_points))
    if count == 0:
        return []

    baseline = baseline_points[:count]
    if count > 8:
        return _greedy_point_distances(baseline, candidate_points)

    best: list[float] | None = None
    for ordering in permutations(candidate_points, count):
        distances = [
            math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))
            for left, right in zip(baseline, ordering, strict=True)
        ]
        if best is None or sum(distances) < sum(best):
            best = distances
    return [round(distance, 6) for distance in (best or [])]


def _greedy_point_distances(
    baseline_points: list[dict[str, Any]], candidate_points: list[dict[str, Any]]
) -> list[float]:
    """Bounded-cost fallback for unusually large smoke-test point sets."""

    remaining = list(candidate_points)
    distances: list[float] = []
    for baseline in baseline_points:
        if not remaining:
            break
        index, distance = min(
            enumerate(remaining),
            key=lambda item: math.hypot(
                float(baseline["x"]) - float(item[1]["x"]),
                float(baseline["y"]) - float(item[1]["y"]),
            ),
        )
        candidate = remaining.pop(index)
        distances.append(
            math.hypot(
                float(baseline["x"]) - float(candidate["x"]),
                float(baseline["y"]) - float(candidate["y"]),
            )
        )
    return [round(distance, 6) for distance in distances]
