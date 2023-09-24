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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    """
    This function calculates the inverse of the sigmoid function for a given input.
    
    Args:
        x (torch.Tensor): Input tensor for which the inverse sigmoid is to be calculated.
    
    Returns:
        torch.Tensor: The inverse sigmoid of the input tensor.
    """
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    """
    This function converts a PIL image to a PyTorch tensor and resizes it to the given resolution.
    
    Args:
        pil_image (PIL.Image): The input image in PIL format.
        resolution (tuple): The desired resolution to resize the image to.
    
    Returns:
        torch.Tensor: The resized image in PyTorch tensor format.
    """
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    This function generates a learning rate decay function for optimization. The learning rate starts
    from lr_init and decays exponentially to lr_final over max_steps. If lr_delay_steps is specified,
    the learning rate is scaled by a smooth function of lr_delay_mult initially.
    
    Args:
        lr_init (float): The initial learning rate.
        lr_final (float): The final learning rate.
        lr_delay_steps (int, optional): The number of steps to delay before starting decay.
        lr_delay_mult (float, optional): The multiplier for the learning rate during the delay period.
        max_steps (int, optional): The total number of steps for learning rate decay.
    
    Returns:
        function: A function that takes the current step as input and returns the learning rate.
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    """
    This function extracts the lower diagonal elements of a 3D tensor and returns them in a 2D tensor.
    
    Args:
        L (torch.Tensor): The input 3D tensor.
    
    Returns:
        torch.Tensor: The 2D tensor containing the lower diagonal elements of the input tensor.
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    """
    This function extracts the lower diagonal elements of a symmetric tensor.
    
    Args:
        sym (torch.Tensor): The input symmetric tensor.
    
    Returns:
        torch.Tensor: The tensor containing the lower diagonal elements of the input tensor.
    """
    return strip_lowerdiag(sym)

def build_rotation(r):
    """
    This function builds a rotation matrix from a quaternion.
    
    Args:
        r (torch.Tensor): The input quaternion.
    
    Returns:
        torch.Tensor: The rotation matrix built from the input quaternion.
    """
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    """
    This function builds a scaling rotation matrix from a scale vector and a quaternion.
    
    Args:
        s (torch.Tensor): The input scale vector.
        r (torch.Tensor): The input quaternion.
    
    Returns:
        torch.Tensor: The scaling rotation matrix built from the input scale vector and quaternion.
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    """
    This function sets the state of the system for reproducibility and controls the verbosity of the output.
    
    Args:
        silent (bool): If True, suppresses the output. If False, allows the output.
    """
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
