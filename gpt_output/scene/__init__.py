start
class Scene:
    """
    The Scene class represents a 3D scene with Gaussian models. It provides methods for loading and saving
    the scene, as well as accessing the training and testing cameras. The scene is initialized with a set of
    parameters, a Gaussian model, an optional iteration to load, a shuffle flag, and a list of resolution scales.
    """

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        Initializes the Scene object with the given parameters, Gaussian model, and optional load iteration.
        If a load iteration is provided, the trained model at that iteration is loaded. The scene's training
        and testing cameras are also loaded based on the provided resolution scales. If shuffle is True, the
        training and testing cameras are shuffled.

        Args:
            args (ModelParams): The parameters for the model.
            gaussians (GaussianModel): The Gaussian model for the scene.
            load_iteration (int, optional): The iteration to load the trained model from. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the training and testing cameras. Defaults to True.
            resolution_scales (list, optional): The scales for the camera resolutions. Defaults to [1.0].
        """

    def save(self, iteration):
        """
        Saves the scene's Gaussian model as a point cloud at the given iteration.

        Args:
            iteration (int): The iteration to save the model at.
        """

    def getTrainCameras(self, scale=1.0):
        """
        Returns the training cameras for the scene at the given scale.

        Args:
            scale (float, optional): The scale for the camera resolution. Defaults to 1.0.

        Returns:
            list: The training cameras for the scene.
        """

    def getTestCameras(self, scale=1.0):
        """
        Returns the testing cameras for the scene at the given scale.

        Args:
            scale (float, optional): The scale for the camera resolution. Defaults to 1.0.

        Returns:
            list: The testing cameras for the scene.
        """
end