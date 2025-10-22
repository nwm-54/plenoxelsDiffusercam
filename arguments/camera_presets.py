import copy
from typing import Dict, List, Literal, Tuple

from arguments import ModelParams, OptimizationParams

MULTIVIEW_INDICES: Dict[Literal[1, 3, 5], Dict[str, List[int]]] = {
    5: {  # 5 views
        "lego": [50, 59, 60, 70, 90],
        "hotdog": [0, 11, 23, 27, 37],
        "chair": [2, 25, 38, 79, 90],
        "drums": [2, 25, 38, 79, 90],
        "ficus": [2, 25, 38, 79, 90],
        "materials": [2, 25, 38, 79, 90],
        "mic": [2, 25, 38, 79, 90],
        "ship": [2, 25, 38, 79, 90],
    },
    3: {  # 3 views
        "lego": [50, 70, 90],
        "hotdog": [0, 23, 37],
        "chair": [2, 79, 90],
        "drums": [25, 38, 90],
        "ficus": [2, 79, 90],
        "materials": [2, 38, 79],
        "mic": [2, 38, 79],
        "ship": [2, 25, 79],
    },
    1: {  # single view
        "lego": [59],
        "hotdog": [2],
        "chair": [2],
        "drums": [3],
        "ficus": [2],
        "materials": [79],
        "mic": [25],
        "ship": [25],
    },
}

BASELINE_PARAMS = {
    "position_lr_init": 0.00016,
    "densify_grad_threshold": 1.5e-05,
    "densification_interval": 50,
    "densify_from_iter": 300,
    "opacity_reset_interval": 1000,
    "lambda_dssim": 0.1,
    "tv_weight": 0.0,
    "tv_unseen_weight": 0.0,
}

BALANCED_WINDOW_PARAMS = {
    "position_lr_init": 0.00024,
    "position_lr_final": 8e-06,
    "densify_grad_threshold": 0.000013,
    "densification_interval": 35,
    "densify_from_iter": 220,
    "densify_until_iter": 2600,
    "opacity_reset_interval": 600,
    "lambda_dssim": 0.16,
    "tv_weight": 0.0025,
    "tv_unseen_weight": 0.0003,
    "opacity_lr": 0.07,
}

TV_MIX_PARAMS = {
    "position_lr_init": 0.00027,
    "position_lr_final": 8e-06,
    "densify_grad_threshold": 0.000011,
    "densification_interval": 30,
    "densify_from_iter": 190,
    "densify_until_iter": 2550,
    "opacity_reset_interval": 540,
    "lambda_dssim": 0.17,
    "tv_weight": 0.0035,
    "tv_unseen_weight": 0.00035,
}

BEST_CAMERA_CONFIG: Dict[str, Dict[str, float]] = {
    "base": BASELINE_PARAMS,
    "iphone": BASELINE_PARAMS,
    "stereo": BALANCED_WINDOW_PARAMS,
    "lightfield": TV_MIX_PARAMS,
    "multiplexing": TV_MIX_PARAMS,
}


def resolve_camera_profile(params: ModelParams, dls: int) -> str:
    """Determine the camera profile key for a given dataset configuration."""
    if params.use_multiplexing:
        return "lightfield" if dls == 12 else "multiplexing"
    if params.use_stereo:
        return "stereo"
    if params.use_iphone:
        return "iphone"
    return "base"


def get_optimization_profile(
    base_opt: OptimizationParams, profile: str
) -> OptimizationParams:
    """
    Produce a copy of ``base_opt`` with preset values applied for the camera profile.

    Keeps the parser-generated namespace intact while adjusting only the tuned keys.
    """
    preset = BEST_CAMERA_CONFIG.get(profile)
    if not preset:
        return base_opt

    tuned_opt = copy.deepcopy(base_opt)
    for key, value in preset.items():
        if hasattr(tuned_opt, key):
            setattr(tuned_opt, key, value)
    return tuned_opt


def apply_profile(
    params: ModelParams, opt: OptimizationParams, dls: int
) -> Tuple[OptimizationParams, str]:
    """Return tuned optimization params and the profile key used."""

    profile = resolve_camera_profile(params, dls)
    tuned_opt = get_optimization_profile(opt, profile)
    return tuned_opt, profile
