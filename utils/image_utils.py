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
    """
    Calculates the Mean Squared Error (MSE) between two images.
    
    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
    
    Returns:
        torch.Tensor: The Mean Squared Error (MSE) between the two input images.
    """
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
    
    Returns:
        torch.Tensor: The Peak Signal-to-Noise Ratio (PSNR) between the two input images.
    """
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
