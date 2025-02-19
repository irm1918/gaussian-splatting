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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    """
    Computes the L1 loss between the network output and the ground truth.
    
    Args:
        network_output (torch.Tensor): The output tensor from the network.
        gt (torch.Tensor): The ground truth tensor.
        
    Returns:
        torch.Tensor: The mean absolute error between the network output and the ground truth.
    """
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    """
    Computes the L2 loss between the network output and the ground truth.
    
    Args:
        network_output (torch.Tensor): The output tensor from the network.
        gt (torch.Tensor): The ground truth tensor.
        
    Returns:
        torch.Tensor: The mean squared error between the network output and the ground truth.
    """
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    """
    Generates a 1D Gaussian window.
    
    Args:
        window_size (int): The size of the window.
        sigma (float): The standard deviation of the Gaussian.
        
    Returns:
        torch.Tensor: A 1D Gaussian window.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Creates a 2D Gaussian window.
    
    Args:
        window_size (int): The size of the window.
        channel (int): The number of channels.
        
    Returns:
        torch.autograd.Variable: A 2D Gaussian window.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
        window_size (int, optional): The size of the Gaussian window. Defaults to 11.
        size_average (bool, optional): Whether to average the SSIM over the image. Defaults to True.
        
    Returns:
        torch.Tensor: The SSIM between the two images.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Helper function to compute the Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
        window (torch.Tensor): The Gaussian window.
        window_size (int): The size of the Gaussian window.
        channel (int): The number of channels.
        size_average (bool, optional): Whether to average the SSIM over the image. Defaults to True.
        
    Returns:
        torch.Tensor: The SSIM between the two images.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

