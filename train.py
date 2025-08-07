from models.promptir_model import PromptIR
from dataset.dataset_dehaze import DehazeDataset
from utils.utils_diffusion import apply_forward_diffusion
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = PromptIR().to(device)
    model.train()

    # Dataset and Dataloader
    train_set = DehazeDataset(train=True)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

    # Loss and Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        total_loss = 0
        for hazy, gt in tqdm(train_loader):
            hazy, gt = hazy.to(device), gt.to(device)

            # Apply forward diffusion to hazy images
            hazy_noisy = apply_forward_diffusion(hazy)

            output = model(hazy_noisy)
            loss = criterion(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(train_loader):.4f}")

        # Save checkpoints
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/promptir_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    train()
