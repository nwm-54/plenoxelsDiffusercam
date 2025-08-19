import torch

from .modules.lpips import LPIPS

_lpips_cache = {}


def lpips(
    x: torch.Tensor, y: torch.Tensor, net_type: str = "alex", version: str = "0.1"
) -> torch.Tensor:
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    cache_key = (net_type, version)

    if cache_key not in _lpips_cache:
        _lpips_cache[cache_key] = LPIPS(net_type=net_type, version=version).to(device)

    criterion = _lpips_cache[cache_key]
    return criterion(x, y)
