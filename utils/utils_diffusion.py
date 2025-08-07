import torch
import torch.nn.functional as F

def apply_forward_diffusion(x, noise_level=0.1):
    noise = torch.randn_like(x) * noise_level
    return x + noise
