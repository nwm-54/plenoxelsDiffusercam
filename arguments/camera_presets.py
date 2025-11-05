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
    "base": {
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30_000,
        "feature_lr": 0.0025,
        "opacity_lr": 0.05,
        "scaling_lr": 0.005,
        "rotation_lr": 0.001,
        "percent_dense": 0.06,
        "lambda_dssim": 0.1,
        "densification_interval": 50,
        "opacity_reset_interval": 1000,
        "densify_from_iter": 300,
        "densify_until_iter": 3_000,
        "densify_grad_threshold": 1.5e-05,
        "tv_weight": 0.0,
        "tv_unseen_weight": 0.0,
    },
    "stereo": {
        "position_lr_init": 2.9e-04,
        "position_lr_final": 2.2e-05,
        "densify_grad_threshold": 1.4e-05,
        "densification_interval": 57,
        "densify_from_iter": 269,
        "densify_until_iter": 2502,
        "opacity_reset_interval": 314,
        "lambda_dssim": 0.2,
        "tv_weight": 2.2e-04,
        "tv_unseen_weight": 2.7e-05,
        "percent_dense": 0.052,
        "opacity_lr": 0.045,
        "scaling_lr": 0.0022,
        "rotation_lr": 1e-03,
    },
    "iphone": {
        "position_lr_init": 2e-04,
        "position_lr_final": 1.5e-05,
        "densify_grad_threshold": 1.8e-05,
        "densification_interval": 28,
        "densify_from_iter": 330,
        "densify_until_iter": 2071,
        "opacity_reset_interval": 337,
        "lambda_dssim": 0.2,
        "tv_weight": 9.7e-04,
        "tv_unseen_weight": 4.1e-05,
        "percent_dense": 0.015,
        "opacity_lr": 0.11,
        "scaling_lr": 0.0057,
        "rotation_lr": 2.3e-04,
    },
    "multiplexing_dls12": {
        "position_lr_init": 4.5e-04,
        "position_lr_final": 2.1e-05,
        "densify_grad_threshold": 6.4e-06,
        "densification_interval": 21,
        "densify_from_iter": 161,
        "densify_until_iter": 2258,
        "opacity_reset_interval": 1192,
        "lambda_dssim": 0.2,
        "tv_weight": 7.6e-03,
        "tv_unseen_weight": 4.3e-04,
        "percent_dense": 0.095,
        "opacity_lr": 0.04,
        "scaling_lr": 0.002,
        "rotation_lr": 7.2e-04,
    },
    "multiplexing_dls20": {
        "position_lr_init": 2.8e-04,
        "densify_grad_threshold": 1.0e-05,
        "densification_interval": 25,
        "densify_from_iter": 150,
        "densify_until_iter": 2400,
        "opacity_reset_interval": 520,
        "lambda_dssim": 0.18,
        "tv_weight": 5.0e-03,
        "tv_unseen_weight": 5.0e-04,
    },
    "multiplexing_midfield": {
        "position_lr_init": 3.4e-04,
        "position_lr_final": 1.3e-05,
        "densify_grad_threshold": 5.7e-06,
        "densification_interval": 29,
        "densify_from_iter": 230,
        "densify_until_iter": 2335,
        "opacity_reset_interval": 1025,
        "lambda_dssim": 0.2,
        "tv_weight": 0.0049,
        "tv_unseen_weight": 7.0e-04,
        "percent_dense": 0.061,
        "opacity_lr": 0.033,
        "scaling_lr": 0.0029,
        "rotation_lr": 6.0e-04,
    },
    "iphone_lowview": {
        "position_lr_init": 9e-05,
        "position_lr_final": 1e-05,
        "densify_grad_threshold": 3.5e-05,
        "densification_interval": 65,
        "densify_from_iter": 347,
        "densify_until_iter": 2417,
        "opacity_reset_interval": 748,
        "lambda_dssim": 0.2,
        "tv_weight": 1.2e-04,
        "tv_unseen_weight": 2.1e-05,
        "percent_dense": 0.062,
        "opacity_lr": 0.096,
        "scaling_lr": 0.0076,
        "rotation_lr": 2.5e-04,
    },
    "stereo_lowview": {
        "position_lr_init": 0.00016,
        "position_lr_final": 1.5e-05,
        "densify_grad_threshold": 3.2e-05,
        "densification_interval": 64,
        "densify_from_iter": 260,
        "densify_until_iter": 2620,
        "opacity_reset_interval": 610,
        "lambda_dssim": 0.2,
        "tv_weight": 5.8e-04,
        "tv_unseen_weight": 2.2e-04,
        "percent_dense": 0.04,
        "opacity_lr": 0.024,
        "scaling_lr": 0.0018,
        "rotation_lr": 2.4e-04,
    },
}

# Best preset for each camera type (based on empirical results)
BEST_CAMERA_PRESET: Dict[str, str] = {
    "base": "base",
    "iphone": "iphone",
    "stereo": "stereo",
    "lightfield_dls12": "multiplexing_dls12",
    "multiplexing_dls20": "multiplexing_dls20",
    "multiplexing_generic": "multiplexing_midfield",
    "iphone_lowview": "iphone_lowview",
    "stereo_lowview": "stereo_lowview",
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
