import copy
from typing import Any, Dict, List, Literal, Tuple

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

# Optimization presets for different training scenarios
OPTIMIZATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "position_lr_init": 0.00016,
        "densify_grad_threshold": 0.000015,
        "densification_interval": 50,
        "densify_from_iter": 300,
        "opacity_reset_interval": 1000,
        "lambda_dssim": 0.1,
        "tv_weight": 0.0,
        "tv_unseen_weight": 0.0,
    },
    "balanced_window": {
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
    },
    "tv_mix": {
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
    },
    "multiplexing_dls20": {
        "position_lr_init": 0.00024,
        "position_lr_final": 6e-06,
        "densify_grad_threshold": 0.000014,
        "densification_interval": 28,
        "densify_from_iter": 200,
        "densify_until_iter": 2900,
        "opacity_reset_interval": 520,
        "lambda_dssim": 0.18,
        "tv_weight": 0.0045,
        "tv_unseen_weight": 0.00055,
        "percent_dense": 0.05,
        "opacity_lr": 0.082,
        "scaling_lr": 0.004,
        "rotation_lr": 0.00085,
    },
    "lowview_regularized": {
        "position_lr_init": 0.00018,
        "position_lr_final": 6e-06,
        "densify_grad_threshold": 0.000022,
        "densification_interval": 45,
        "densify_from_iter": 220,
        "densify_until_iter": 2400,
        "opacity_reset_interval": 750,
        "percent_dense": 0.045,
        "lambda_dssim": 0.12,
        "tv_weight": 0.0035,
        "tv_unseen_weight": 0.00045,
        "opacity_lr": 0.08,
        "scaling_lr": 0.0045,
        "rotation_lr": 0.0008,
    },
}

# Best preset for each camera type (based on empirical results)
BEST_CAMERA_PRESET: Dict[str, str] = {
    "base": "baseline",
    "iphone": "balanced_window",
    "stereo": "balanced_window",
    "lightfield_dls12": "tv_mix",
    "multiplexing_dls20": "multiplexing_dls20",
    "multiplexing_generic": "tv_mix",
    "iphone_lowview": "lowview_regularized",
    "stereo_lowview": "lowview_regularized",
}

LOW_VIEW_MAX_IMAGES = 3


def resolve_camera_profile(params: ModelParams, dls: int) -> str:
    """Determine the camera profile key for a given dataset configuration."""
    low_view = params.n_train_images <= LOW_VIEW_MAX_IMAGES
    if params.use_multiplexing:
        if dls == 12:
            return "lightfield_dls12"
        if dls >= 20:
            return "multiplexing_dls20"
        return "multiplexing_generic"
    if params.use_stereo:
        return "stereo_lowview" if low_view else "stereo"
    if params.use_iphone:
        return "iphone_lowview" if low_view else "iphone"
    return "base"


def get_optimization_profile(
    base_opt: OptimizationParams, profile: str
) -> OptimizationParams:
    """
    Produce a copy of base_opt with preset values applied for the camera profile.

    Args:
        base_opt: Base optimization parameters from argument parser
        profile: Camera profile key (base, stereo, iphone, multiplexing, lightfield)

    Returns:
        OptimizationParams with preset values applied
    """
    preset_name = BEST_CAMERA_PRESET.get(profile)
    if not preset_name or preset_name not in OPTIMIZATION_PRESETS:
        return base_opt

    preset_params = OPTIMIZATION_PRESETS[preset_name]
    tuned_opt = copy.deepcopy(base_opt)

    for key, value in preset_params.items():
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
