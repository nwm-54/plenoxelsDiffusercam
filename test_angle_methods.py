#!/usr/bin/env python3

import numpy as np
from scene.scene_utils import solve_offset_for_angle

def test_angle_methods():
    # Example vectors from the actual run
    base_vec = np.array([0.1, 0.2, -0.5])  # Example base vector
    axis = np.array([1.0, 0.0, 0.0])       # X axis
    target_angle = 20.0

    print(f"Target angle: {target_angle:.6f}°")
    print(f"Base vector: {base_vec}")
    print(f"Axis: {axis}")
    print()

    offset = solve_offset_for_angle(base_vec, axis, target_angle)
    print(f"Unified offset: {offset:.10f}")

    def compute_angle(base_vec, axis, offset):
        axis_unit = axis / np.linalg.norm(axis)
        parallel = float(np.dot(base_vec, axis_unit))
        perp_vec = base_vec - axis_unit * parallel
        perp_sq = float(np.dot(perp_vec, perp_vec))

        left_sq = (parallel - offset) ** 2 + perp_sq
        right_sq = (parallel + offset) ** 2 + perp_sq
        denom = np.sqrt(left_sq * right_sq)
        numerator = parallel * parallel - offset * offset + perp_sq
        cos_theta = np.clip(numerator / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    angle_result = compute_angle(base_vec, axis, offset)
    print(f"Resulting angle: {angle_result:.10f}°")
    print(f"Error from target: {abs(angle_result - target_angle):.2e}°")

    # Extended test with coupled shift (iphone layout)
    orth_axis = np.array([0.0, 1.0, 0.0])
    ratio = 1.0
    offset_ratio = solve_offset_for_angle(
        base_vec,
        axis,
        target_angle,
        orth_axis=orth_axis,
        orth_ratio=ratio,
    )
    print(f"Coupled-ratio offset: {offset_ratio:.10f}")

    def compute_angle_with_ratio(base_vec, axis, orth_axis, offset, ratio):
        axis_unit = axis / np.linalg.norm(axis)
        orth = orth_axis / np.linalg.norm(orth_axis)
        orth -= np.dot(orth, axis_unit) * axis_unit
        orth /= np.linalg.norm(orth)
        shift = orth * (ratio * offset)
        left_vec = base_vec - shift - axis_unit * offset
        right_vec = base_vec - shift + axis_unit * offset
        left_norm = np.linalg.norm(left_vec)
        right_norm = np.linalg.norm(right_vec)
        cos_theta = np.clip(
            np.dot(left_vec, right_vec) / (left_norm * right_norm),
            -1.0,
            1.0,
        )
        return float(np.degrees(np.arccos(cos_theta)))

    angle_ratio = compute_angle_with_ratio(
        base_vec,
        axis,
        orth_axis,
        offset_ratio,
        ratio,
    )
    print(f"Resulting angle (coupled): {angle_ratio:.10f}°")
    print(f"Error from target (coupled): {abs(angle_ratio - target_angle):.2e}°")

if __name__ == "__main__":
    test_angle_methods()
