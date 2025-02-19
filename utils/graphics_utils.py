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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    """
    A class used to represent a basic point cloud.
    
    Attributes:
        points (np.array): The points in the point cloud.
        colors (np.array): The colors of the points in the point cloud.
        normals (np.array): The normals of the points in the point cloud.
    """
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    """
    Transforms the given points using the provided transformation matrix.
    
    Args:
        points (torch.Tensor): The points to be transformed.
        transf_matrix (torch.Tensor): The transformation matrix.
    
    Returns:
        torch.Tensor: The transformed points.
    """
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    """
    Computes the world-to-view transformation matrix.
    
    Args:
        R (np.array): The rotation matrix.
        t (np.array): The translation vector.
    
    Returns:
        np.array: The world-to-view transformation matrix.
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    Computes the world-to-view transformation matrix with additional translation and scaling.
    
    Args:
        R (np.array): The rotation matrix.
        t (np.array): The translation vector.
        translate (np.array, optional): The additional translation vector. Defaults to np.array([.0, .0, .0]).
        scale (float, optional): The scaling factor. Defaults to 1.0.
    
    Returns:
        np.array: The world-to-view transformation matrix.
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    Computes the projection matrix.
    
    Args:
        znear (float): The near clipping plane distance.
        zfar (float): The far clipping plane distance.
        fovX (float): The horizontal field of view in radians.
        fovY (float): The vertical field of view in radians.
    
    Returns:
        torch.Tensor: The projection matrix.
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    """
    Converts field of view to focal length.
    
    Args:
        fov (float): The field of view in radians.
        pixels (int): The number of pixels in the image.
    
    Returns:
        float: The focal length.
    """
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    """
    Converts focal length to field of view.
    
    Args:
        focal (float): The focal length.
        pixels (int): The number of pixels in the image.
    
    Returns:
        float: The field of view in radians.
    """
    return 2*math.atan(pixels/(2*focal))