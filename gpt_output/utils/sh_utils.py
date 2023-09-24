start

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions using hardcoded SH polynomials.
    
    This function works with torch/np/jnp and can have 0 or more batch dimensions.
    
    Args:
        deg (int): SH degree. Currently, 0-3 are supported.
        sh (jnp.ndarray): SH coefficients with shape [..., C, (deg + 1) ** 2].
        dirs (jnp.ndarray): Unit directions with shape [..., 3].
    
    Returns:
        jnp.ndarray: The result of the spherical harmonics evaluation with shape [..., C].
    """
    

def RGB2SH(rgb):
    """
    Convert RGB values to spherical harmonics (SH) representation.
    
    This function subtracts 0.5 from the input RGB values and divides the result by the constant C0.
    
    Args:
        rgb (torch.Tensor): Input RGB values.
    
    Returns:
        torch.Tensor: The SH representation of the input RGB values.
    """
    

def SH2RGB(sh):
    """
    Convert spherical harmonics (SH) representation to RGB values.
    
    This function multiplies the input SH values by the constant C0 and adds 0.5 to the result.
    
    Args:
        sh (torch.Tensor): Input SH representation.
    
    Returns:
        torch.Tensor: The RGB values of the input SH representation.
    """
    
end