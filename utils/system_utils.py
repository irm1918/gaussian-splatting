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

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    """
    Creates a directory. This function is equivalent to using mkdir -p on the command line.
    
    Args:
        folder_path (str): The path of the directory to be created.
        
    Raises:
        OSError: If the creation of the directory fails due to system-related errors.
    """
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    """
    Searches for the maximum iteration number in a given folder. This function assumes that the 
    iteration number is the last element in the filename, separated by underscores.
    
    Args:
        folder (str): The path of the directory to be searched.
        
    Returns:
        int: The maximum iteration number found in the directory.
    """
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
