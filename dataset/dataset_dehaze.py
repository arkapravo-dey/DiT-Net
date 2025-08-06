# dataset_dehaze.py: Dataset loader for hazy-clean image pairs

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DehazeDataset(Dataset):
    def __init__(self, root='dataset/train', train=True):
        self.train = train
        self.root = root if train else root.replace('train', 'test')
        self.hazy_dir = os.path.join(self.root, 'hazy')
        self.clean_dir = os.path.join(self.root, 'clean')
        self.filenames = sorted(os.listdir(self.hazy_dir))

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.filenames[idx])
        clean_path = os.path.join(self.clean_dir, self.filenames[idx])

        hazy = self.transform(Image.open(hazy_path).convert('RGB'))
        clean = self.transform(Image.open(clean_path).convert('RGB'))

        return hazy, clean
