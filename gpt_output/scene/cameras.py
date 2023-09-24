start

class Camera(nn.Module):
    """
    The Camera class is a PyTorch module that represents a camera in the 3D scene. It includes 
    parameters such as the camera's position, orientation, field of view, and image data. The class 
    also handles the transformation from world coordinates to view coordinates and the projection 
    from view coordinates to image coordinates.

    Args:
        colmap_id (int): The ID of the camera in the COLMAP dataset.
        R (torch.Tensor): The rotation matrix of the camera.
        T (torch.Tensor): The translation vector of the camera.
        FoVx (float): The horizontal field of view of the camera.
        FoVy (float): The vertical field of view of the camera.
        image (torch.Tensor): The image captured by the camera.
        gt_alpha_mask (torch.Tensor): The ground truth alpha mask for the image.
        image_name (str): The name of the image file.
        uid (int): A unique identifier for the camera.
        trans (np.array): A translation vector applied to the camera position.
        scale (float): A scaling factor applied to the camera position.
        data_device (str): The device where the data will be processed ("cuda" or "cpu").
    """

class MiniCam:
    """
    The MiniCam class represents a simplified version of the Camera class, containing only the 
    essential parameters needed for rendering. It includes parameters such as the camera's field 
    of view, image dimensions, and transformation matrices.

    Args:
        width (int): The width of the image captured by the camera.
        height (int): The height of the image captured by the camera.
        fovy (float): The vertical field of view of the camera.
        fovx (float): The horizontal field of view of the camera.
        znear (float): The near clipping plane of the camera.
        zfar (float): The far clipping plane of the camera.
        world_view_transform (torch.Tensor): The transformation matrix from world coordinates to 
                                             view coordinates.
        full_proj_transform (torch.Tensor): The full projection transformation matrix from world 
                                            coordinates to image coordinates.
    """

end