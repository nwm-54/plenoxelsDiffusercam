#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def heteroscedastic_noise(img, lambda_read=0.03, lambda_shot=0.03):
    """Apply heteroscedastic noise to image"""
    noise = torch.randn_like(img) * (lambda_read + lambda_shot * img)
    return torch.clamp(img + noise, 0, 1)

class Quantize14bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """Quantize a tensor to 14-bit"""
        q = (1 << 14) - 1
        return torch.round(x * q) / q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.clone()

def quantize_14bit(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to 14-bit"""
    return Quantize14bit.apply(x)