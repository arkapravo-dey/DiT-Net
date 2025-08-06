# test.py: Evaluation script for PromptIR with PSNR & SSIM

import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F

from models.promptir_model import PromptIR
from dataset.dataset_dehaze import DehazeDataset

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0

def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIR().to(device)
    model.load_state_dict(torch.load("checkpoints/promptir_epoch_50.pth"))
    model.eval()

    test_set = DehazeDataset(train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    os.makedirs("results", exist_ok=True)

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.no_grad():
        for i, (hazy, clean) in enumerate(test_loader):
            hazy, clean = hazy.to(device), clean.to(device)

            output = model(hazy).clamp_(-1, 1)

            # Save image
            save_image(output, f"results/dehazed_{i}.png")

            # Rescale [-1, 1] -> [0, 1]
            output_img = output * 0.5 + 0.5
            clean_img = clean * 0.5 + 0.5

            # PSNR
            mse = F.mse_loss(output_img, clean_img, reduction='none').mean((1, 2, 3))
            psnr = 10 * torch.log10(1 / (mse + 1e-8))
            psnr_meter.update(psnr.item())

            # SSIM
            out_np = (output_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            tgt_np = (clean_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ssim = calculate_ssim(out_np, tgt_np)
            ssim_meter.update(ssim)

    print(f"✅ Average PSNR: {psnr_meter.avg:.2f} dB")
    print(f"✅ Average SSIM: {ssim_meter.avg:.4f}")

if __name__ == '__main__':
    test()
