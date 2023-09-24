start

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
"""
Namedtuple to represent a camera model.

Attributes:
    model_id (int): Unique identifier for the camera model.
    model_name (str): Name of the camera model.
    num_params (int): Number of parameters for the camera model.
"""

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
"""
Namedtuple to represent a camera.

Attributes:
    id (int): Unique identifier for the camera.
    model (str): Camera model.
    width (int): Width of the camera's image.
    height (int): Height of the camera's image.
    params (list): Parameters of the camera model.
"""

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
"""
Namedtuple to represent an image.

Attributes:
    id (int): Unique identifier for the image.
    qvec (list): Quaternion vector representing the image's rotation.
    tvec (list): Translation vector representing the image's position.
    camera_id (int): Identifier of the camera that captured the image.
    name (str): Name of the image.
    xys (list): List of 2D points in the image.
    point3D_ids (list): List of 3D point identifiers corresponding to the 2D points.
"""

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
"""
Namedtuple to represent a 3D point.

Attributes:
    id (int): Unique identifier for the 3D point.
    xyz (list): Coordinates of the 3D point.
    rgb (list): RGB color values of the 3D point.
    error (float): Reprojection error of the 3D point.
    image_ids (list): Identifiers of images where the 3D point is visible.
    point2D_idxs (list): Indices of the corresponding 2D points in the images.
"""

def qvec2rotmat(qvec):
    """
    Converts a quaternion vector to a rotation matrix.

    Args:
        qvec (list): Quaternion vector.

    Returns:
        np.array: 3x3 rotation matrix.
    """

def rotmat2qvec(R):
    """
    Converts a rotation matrix to a quaternion vector.

    Args:
        R (np.array): 3x3 rotation matrix.

    Returns:
        np.array: Quaternion vector.
    """

class Image(BaseImage):
    def qvec2rotmat(self):
        """
        Converts the image's quaternion vector to a rotation matrix.

        Returns:
            np.array: 3x3 rotation matrix.
        """

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """
    Reads and unpacks the next bytes from a binary file.

    Args:
        fid (file): Binary file.
        num_bytes (int): Number of bytes to read.
        format_char_sequence (str): Format characters for struct.unpack.
        endian_character (str, optional): Endian character for struct.unpack. Defaults to "<".

    Returns:
        tuple: Unpacked values.
    """

def read_points3D_text(path):
    """
    Reads 3D points from a text file.

    Args:
        path (str): Path to the text file.

    Returns:
        tuple: Arrays of 3D coordinates, RGB colors, and errors.
    """

def read_points3D_binary(path_to_model_file):
    """
    Reads 3D points from a binary file.

    Args:
        path_to_model_file (str): Path to the binary file.

    Returns:
        tuple: Arrays of 3D coordinates, RGB colors, and errors.
    """

def read_intrinsics_text(path):
    """
    Reads camera intrinsics from a text file.

    Args:
        path (str): Path to the text file.

    Returns:
        dict: Dictionary of cameras.
    """

def read_extrinsics_binary(path_to_model_file):
    """
    Reads camera extrinsics from a binary file.

    Args:
        path_to_model_file (str): Path to the binary file.

    Returns:
        dict: Dictionary of images.
    """

def read_intrinsics_binary(path_to_model_file):
    """
    Reads camera intrinsics from a binary file.

    Args:
        path_to_model_file (str): Path to the binary file.

    Returns:
        dict: Dictionary of cameras.
    """

def read_extrinsics_text(path):
    """
    Reads camera extrinsics from a text file.

    Args:
        path (str): Path to the text file.

    Returns:
        dict: Dictionary of images.
    """

def read_colmap_bin_array(path):
    """
    Reads a binary array from a COLMAP file.

    Args:
        path (str): Path to the COLMAP file.

    Returns:
        np.array: Array with the floating point values in the file.
    """

end