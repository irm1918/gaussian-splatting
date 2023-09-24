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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    """
    This class is a placeholder for grouping parameters.
    """
    pass

class ParamGroup:
    """
    This class is used to create a group of parameters for the ArgumentParser. It allows for the
    addition of arguments with shorthand and handles the extraction of argument values.
    """
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        """
        Initializes the ParamGroup with a given parser, name, and fill_none flag.
        
        Args:
            parser (ArgumentParser): The parser to which the argument group is added.
            name (str): The name of the argument group.
            fill_none (bool, optional): If True, all values are filled with None. Defaults to False.
        """
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        """
        Extracts the arguments from the given args.
        
        Args:
            args: The arguments from which to extract.
        
        Returns:
            GroupParams: A GroupParams object with the extracted arguments.
        """
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    """
    This class is a subclass of ParamGroup, specifically for model parameters. It includes parameters
    related to the spherical harmonics degree, source path, model path, images, resolution, background,
    data device, and evaluation mode.
    """
    def __init__(self, parser, sentinel=False):
        """
        Initializes the ModelParams with a given parser and sentinel flag.
        
        Args:
            parser: The parser to which the argument group is added.
            sentinel (bool, optional): A flag used to control the initialization. Defaults to False.
        """
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        """
        Extracts the arguments from the given args and converts the source path to an absolute path.
        
        Args:
            args: The arguments from which to extract.
        
        Returns:
            GroupParams: A GroupParams object with the extracted arguments.
        """
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    """
    This class is a subclass of ParamGroup, specifically for pipeline parameters. It includes parameters
    related to the conversion of spherical harmonics, computation of 3D covariance, and debug mode.
    """
    def __init__(self, parser):
        """
        Initializes the PipelineParams with a given parser.
        
        Args:
            parser: The parser to which the argument group is added.
        """
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    """
    This class is a subclass of ParamGroup, specifically for optimization parameters. It includes parameters
    related to the iterations, learning rates, density, lambda, densification interval, opacity reset interval,
    densification iteration range, and densification gradient threshold.
    """
    def __init__(self, parser):
        """
        Initializes the OptimizationParams with a given parser.
        
        Args:
            parser: The parser to which the argument group is added.
        """
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    """
    This function combines the command line arguments and the arguments from a configuration file.
    
    Args:
        parser (ArgumentParser): The parser from which the arguments are extracted.
    
    Returns:
        Namespace: A Namespace object with the combined arguments.
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
