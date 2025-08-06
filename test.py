# test.py: Evaluation script for PromptIR

from models.promptir_model import PromptIR
from dataset.dataset_dehaze import DehazeDataset
from torch.utils.data import DataLoader
import torch
import os
from torchvision.utils import save_image

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PromptIR().to(device)
    model.load_state_dict(torch.load("checkpoints/promptir_epoch_50.pth"))
    model.eval()

    test_set = DehazeDataset(train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    os.makedirs("results", exist_ok=True)
    with torch.no_grad():
        for i, (hazy, _) in enumerate(test_loader):
            hazy = hazy.to(device)
            output = model(hazy)
            save_image(output, f"results/dehazed_{i}.png")

if __name__ == '__main__':
    test()
