import torch
import torch.nn.functional as F

def apply_forward_diffusion(x, noise_level=0.1):
    """
    Applies simple forward noise (Gaussian) as placeholder for diffusion process
    """
    noise = torch.randn_like(x) * noise_level
    return x + noise
