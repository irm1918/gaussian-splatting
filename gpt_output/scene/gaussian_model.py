start
class GaussianModel:
    """
    This class represents the Gaussian Model used for real-time rendering of radiance fields using 3D
    Gaussian splatting. It includes methods for setting up functions, capturing and restoring model states,
    getting model parameters, and manipulating point cloud data.
    """

    def setup_functions(self):
        """
        Sets up the necessary functions for the Gaussian Model. This includes scaling, rotation, and opacity
        activations, as well as a function to build covariance from scaling and rotation.
        """

    def __init__(self, sh_degree: int):
        """
        Initializes the Gaussian Model with the given spherical harmonics degree. It also initializes the model
        parameters and sets up the necessary functions.
        
        Args:
            sh_degree (int): The degree of spherical harmonics to be used in the model.
        """

    def capture(self):
        """
        Captures the current state of the Gaussian Model. This includes the active spherical harmonics degree,
        model parameters, optimizer state, and spatial learning rate scale.
        
        Returns:
            tuple: A tuple containing the current state of the Gaussian Model.
        """

    def restore(self, model_args, training_args):
        """
        Restores the Gaussian Model to a previous state using the given model and training arguments. It also
        sets up the training environment using the given training arguments.
        
        Args:
            model_args (tuple): A tuple containing the model arguments to restore the Gaussian Model state.
            training_args (object): An object containing the training arguments to set up the training environment.
        """

    @property
    def get_scaling(self):
        """
        Returns the scaling of the Gaussian Model after applying the scaling activation function.
        
        Returns:
            tensor: A tensor representing the scaling of the Gaussian Model.
        """

    @property
    def get_rotation(self):
        """
        Returns the rotation of the Gaussian Model after applying the rotation activation function.
        
        Returns:
            tensor: A tensor representing the rotation of the Gaussian Model.
        """

    @property
    def get_xyz(self):
        """
        Returns the xyz coordinates of the Gaussian Model.
        
        Returns:
            tensor: A tensor representing the xyz coordinates of the Gaussian Model.
        """

    @property
    def get_features(self):
        """
        Returns the features of the Gaussian Model by concatenating the DC and rest features.
        
        Returns:
            tensor: A tensor representing the features of the Gaussian Model.
        """

    @property
    def get_opacity(self):
        """
        Returns the opacity of the Gaussian Model after applying the opacity activation function.
        
        Returns:
            tensor: A tensor representing the opacity of the Gaussian Model.
        """

    def get_covariance(self, scaling_modifier=1):
        """
        Returns the covariance of the Gaussian Model by applying the covariance activation function on the scaling
        and rotation of the model.
        
        Args:
            scaling_modifier (int, optional): A modifier for the scaling of the Gaussian Model. Defaults to 1.
        
        Returns:
            tensor: A tensor representing the covariance of the Gaussian Model.
        """

    def oneupSHdegree(self):
        """
        Increases the active spherical harmonics degree of the Gaussian Model by one, if it is less than the maximum
        spherical harmonics degree.
        """

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Creates the Gaussian Model from a given point cloud data (pcd). It also sets the spatial learning rate scale
        of the model.
        
        Args:
            pcd (BasicPointCloud): The point cloud data to create the Gaussian Model from.
            spatial_lr_scale (float): The spatial learning rate scale to set for the Gaussian Model.
        """

    def training_setup(self, training_args):
        """
        Sets up the training environment for the Gaussian Model using the given training arguments.
        
        Args:
            training_args (object): An object containing the training arguments to set up the training environment.
        """

    def update_learning_rate(self, iteration):
        """
        Updates the learning rate of the Gaussian Model's optimizer for the given iteration.
        
        Args:
            iteration (int): The current iteration to update the learning rate for.
        
        Returns:
            float: The updated learning rate.
        """

    def construct_list_of_attributes(self):
        """
        Constructs a list of attributes for the Gaussian Model. This includes the xyz coordinates, features, opacity,
        scaling, and rotation.
        
        Returns:
            list: A list of attributes for the Gaussian Model.
        """

    def save_ply(self, path):
        """
        Saves the Gaussian Model as a .ply file at the given path.
        
        Args:
            path (str): The path to save the .ply file at.
        """

    def reset_opacity(self):
        """
        Resets the opacity of the Gaussian Model by replacing the opacity tensor in the optimizer with a new one.
        """

    def load_ply(self, path):
        """
        Loads the Gaussian Model from a .ply file at the given path.
        
        Args:
            path (str): The path to load the .ply file from.
        """

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        Replaces a tensor in the Gaussian Model's optimizer with a new one. The new tensor is also made optimizable.
        
        Args:
            tensor (tensor): The new tensor to replace the existing one with.
            name (str): The name of the tensor to replace.
        
        Returns:
            dict: A dictionary containing the optimizable tensors of the Gaussian Model.
        """

    def _prune_optimizer(self, mask):
        """
        Prunes the Gaussian Model's optimizer using the given mask. This involves replacing the existing tensors in
        the optimizer with new ones that only contain the valid points.
        
        Args:
            mask (tensor): A mask indicating the valid points in the Gaussian Model.
        
        Returns:
            dict: A dictionary containing the optimizable tensors of the Gaussian Model.
        """

    def prune_points(self, mask):
        """
        Prunes the points of the Gaussian Model using the given mask. This involves replacing the existing tensors in
        the model with new ones that only contain the valid points.
        
        Args:
            mask (tensor): A mask indicating the valid points in the Gaussian Model.
        """
end

start
def cat_tensors_to_optimizer(self, tensors_dict):
    """
    Concatenates tensors to the optimizer. This function is used to add new tensors to the optimizer
    during the densification process. The tensors are added to the optimizer's parameter groups.

    Args:
        tensors_dict (dict): A dictionary containing the tensors to be added to the optimizer.

    Returns:
        dict: A dictionary containing the optimizable tensors.
    """

def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
    """
    Updates the model's attributes after the densification process. This function is called after
    new points have been added to the model during the densification process.

    Args:
        new_xyz (torch.Tensor): The new xyz coordinates.
        new_features_dc (torch.Tensor): The new direct color features.
        new_features_rest (torch.Tensor): The new rest features.
        new_opacities (torch.Tensor): The new opacities.
        new_scaling (torch.Tensor): The new scaling factors.
        new_rotation (torch.Tensor): The new rotation parameters.
    """

def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
    """
    Densifies the model by splitting points that satisfy certain conditions. This function is part
    of the densification process, where points with a gradient above a certain threshold are split
    into multiple points.

    Args:
        grads (torch.Tensor): The gradients of the points.
        grad_threshold (float): The gradient threshold for splitting points.
        scene_extent (float): The extent of the scene.
        N (int, optional): The number of points to split into. Defaults to 2.
    """

def densify_and_clone(self, grads, grad_threshold, scene_extent):
    """
    Densifies the model by cloning points that satisfy certain conditions. This function is part
    of the densification process, where points with a gradient above a certain threshold are cloned.

    Args:
        grads (torch.Tensor): The gradients of the points.
        grad_threshold (float): The gradient threshold for cloning points.
        scene_extent (float): The extent of the scene.
    """

def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
    """
    Densifies and prunes the model. This function is part of the densification process, where points
    are added and removed based on certain conditions.

    Args:
        max_grad (float): The maximum gradient for densification.
        min_opacity (float): The minimum opacity for pruning.
        extent (float): The extent of the scene.
        max_screen_size (float): The maximum screen size for pruning.
    """

def add_densification_stats(self, viewspace_point_tensor, update_filter):
    """
    Updates the densification statistics. This function is called during the densification process
    to update the gradient accumulation and the denominator used for normalization.

    Args:
        viewspace_point_tensor (torch.Tensor): The viewspace point tensor.
        update_filter (torch.Tensor): The update filter.
    """
end

