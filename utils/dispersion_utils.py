from __future__ import annotations

from typing import Optional

import numpy as np


def find_max_min_dispersion_subset(
    points: np.ndarray, k: int, initial_point_index: Optional[int] = None
) -> np.ndarray:
    """Find a near-optimal subset of k points that maximizes minimum pairwise distance."""
    if k < 1:
        return np.array([], dtype=int)
    n_points = len(points)
    if k >= n_points:
        return np.arange(n_points, dtype=int)

    selected_indices = np.zeros(k, dtype=int)
    rng = np.random.default_rng(seed=42)
    selected_indices[0] = (
        initial_point_index
        if initial_point_index is not None
        else int(rng.integers(n_points))
    )

    dists = np.linalg.norm(points - points[selected_indices[0]], axis=1)
    for i in range(1, k):
        farthest_idx = int(np.argmax(dists))
        selected_indices[i] = farthest_idx
        new_dists = np.linalg.norm(points - points[farthest_idx], axis=1)
        dists = np.minimum(dists, new_dists)

    for _ in range(10):
        current_subset = set(int(idx) for idx in selected_indices)

        sub_dist_matrix = np.linalg.norm(
            points[selected_indices, None] - points[None, selected_indices], axis=-1
        )
        np.fill_diagonal(sub_dist_matrix, np.inf)
        min_dist = float(sub_dist_matrix.min())

        made_swap = False
        for i in range(k):
            for p_out_idx in range(n_points):
                if p_out_idx in current_subset:
                    continue

                temp_indices = np.array(selected_indices, copy=True)
                temp_indices[i] = p_out_idx
                new_sub_dist_matrix = np.linalg.norm(
                    points[temp_indices, None] - points[None, temp_indices], axis=-1
                )
                np.fill_diagonal(new_sub_dist_matrix, np.inf)
                new_min_dist = float(new_sub_dist_matrix.min())

                if new_min_dist > min_dist:
                    selected_indices = temp_indices
                    min_dist = new_min_dist
                    made_swap = True
                    break
            if made_swap:
                break
        if not made_swap:
            break

    return selected_indices
