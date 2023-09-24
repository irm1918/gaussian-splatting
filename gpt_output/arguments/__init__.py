start
class GroupParams:
    """This class is a placeholder for grouping parameters."""

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

    def extract(self, args):
        """
        Extracts the arguments from the given args.

        Args:
            args: The arguments from which to extract.

        Returns:
            GroupParams: A GroupParams object with the extracted arguments.
        """

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

    def extract(self, args):
        """
        Extracts the arguments from the given args and converts the source path to an absolute path.

        Args:
            args: The arguments from which to extract.

        Returns:
            GroupParams: A GroupParams object with the extracted arguments.
        """

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

def get_combined_args(parser : ArgumentParser):
    """
    This function combines the command line arguments and the arguments from a configuration file.

    Args:
        parser (ArgumentParser): The parser from which the arguments are extracted.

    Returns:
        Namespace: A Namespace object with the combined arguments.
    """
end