start
class CameraInfo(NamedTuple):
    """
    A NamedTuple representing the camera information.

    Attributes:
        uid (int): Unique identifier for the camera.
        R (np.array): Rotation matrix of the camera.
        T (np.array): Translation vector of the camera.
        FovY (np.array): Field of view in the Y direction.
        FovX (np.array): Field of view in the X direction.
        image (np.array): Image captured by the camera.
        image_path (str): Path to the image file.
        image_name (str): Name of the image file.
        width (int): Width of the image.
        height (int): Height of the image.
    """

class SceneInfo(NamedTuple):
    """
    A NamedTuple representing the scene information.

    Attributes:
        point_cloud (BasicPointCloud): Point cloud representing the scene.
        train_cameras (list): List of cameras used for training.
        test_cameras (list): List of cameras used for testing.
        nerf_normalization (dict): Normalization parameters for NeRF.
        ply_path (str): Path to the .ply file representing the scene.
    """

def getNerfppNorm(cam_info):
    """
    Computes the center and diagonal of the bounding box enclosing the camera centers.

    Args:
        cam_info (list): List of CameraInfo objects.

    Returns:
        dict: Dictionary containing the translation vector and radius of the bounding sphere.
    """

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    """
    Reads camera information from COLMAP extrinsics and intrinsics.

    Args:
        cam_extrinsics (dict): Dictionary of camera extrinsics.
        cam_intrinsics (dict): Dictionary of camera intrinsics.
        images_folder (str): Path to the folder containing the images.

    Returns:
        list: List of CameraInfo objects.
    """

def fetchPly(path):
    """
    Reads a .ply file and returns a BasicPointCloud object.

    Args:
        path (str): Path to the .ply file.

    Returns:
        BasicPointCloud: Point cloud representing the scene.
    """

def storePly(path, xyz, rgb):
    """
    Stores a point cloud as a .ply file.

    Args:
        path (str): Path to the .ply file.
        xyz (np.array): 3D coordinates of the points.
        rgb (np.array): RGB colors of the points.
    """

def readColmapSceneInfo(path, images, eval, llffhold=8):
    """
    Reads scene information from COLMAP files.

    Args:
        path (str): Path to the COLMAP files.
        images (str): Path to the images.
        eval (bool): Whether to split the cameras into training and testing sets.
        llffhold (int): Interval for selecting test cameras.

    Returns:
        SceneInfo: Scene information.
    """

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    """
    Reads camera information from a transforms file.

    Args:
        path (str): Path to the transforms file.
        transformsfile (str): Name of the transforms file.
        white_background (bool): Whether the images have a white background.
        extension (str): Extension of the image files.

    Returns:
        list: List of CameraInfo objects.
    """

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    """
    Reads scene information from NeRF synthetic data.

    Args:
        path (str): Path to the NeRF synthetic data.
        white_background (bool): Whether the images have a white background.
        eval (bool): Whether to split the cameras into training and testing sets.
        extension (str): Extension of the image files.

    Returns:
        SceneInfo: Scene information.
    """

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
end