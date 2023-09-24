start
def readImages(renders_dir, gt_dir):
    """
    Reads images from the given directories and converts them into tensors.

    Args:
        renders_dir (str): The directory containing the rendered images.
        gt_dir (str): The directory containing the ground truth images.

    Returns:
        list: A list of tensors representing the rendered images.
        list: A list of tensors representing the ground truth images.
        list: A list of filenames of the images.
    """

def evaluate(model_paths):
    """
    Evaluates the performance of the models on the test data.

    This function computes the SSIM, PSNR, and LPIPS metrics for each model on the test data. 
    The results are stored in dictionaries and saved as JSON files.

    Args:
        model_paths (list): A list of paths to the directories containing the models to be evaluated.
    """
end