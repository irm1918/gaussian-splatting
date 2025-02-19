from collections import OrderedDict

import torch


def normalize_activation(x, eps=1e-10):
    """
    Normalizes the activation of the input tensor.
    
    This function normalizes the activation of the input tensor by dividing each element by the
    square root of the sum of squares of all elements in the tensor. A small constant is added to
    the denominator to prevent division by zero.
    
    Args:
        x (torch.Tensor): The input tensor whose activation is to be normalized.
        eps (float, optional): A small constant added to the denominator to prevent division by 
            zero. Defaults to 1e-10.
    
    Returns:
        torch.Tensor: The input tensor with normalized activation.
    """
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    """
    Downloads and returns the state dictionary of a specified network type and version.
    
    This function builds a URL to download the state dictionary of a specified network type and
    version from a GitHub repository. It then downloads the state dictionary, renames the keys,
    and returns the new state dictionary.
    
    Args:
        net_type (str, optional): The type of the network for which the state dictionary is to be
            downloaded. Defaults to 'alex'.
        version (str, optional): The version of the network for which the state dictionary is to be
            downloaded. Defaults to '0.1'.
    
    Returns:
        collections.OrderedDict: The downloaded state dictionary with renamed keys.
    """
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
