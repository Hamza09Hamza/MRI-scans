import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T

class BrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the manifest csv.
            root_dir (string): Directory with all the images/masks.
            transform (callable, optional): Optional transform to be applied.
        """
        self.data_info = pd.read_csv(csv_file)
        # We only want slices that HAVE tumors for our first training run
        self.data_info = self.data_info[self.data_info['Has_Tumor'] == True].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)
    

    def __getitem__(self, idx):
        # Using column names is much safer than index 2 or 3
        img_rel_path = self.data_info.iloc[idx]['Image_Path']
        mask_rel_path = self.data_info.iloc[idx]['Mask_Path']
        
        img_path = os.path.join(self.root_dir, img_rel_path)
        mask_path = os.path.join(self.root_dir, mask_rel_path)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            # FIX: Ensure image and mask get the same random transformations
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        mask = torch.as_tensor(np.array(mask) > 0, dtype=torch.float32)
        return image, mask
    
# Let's define the preparation for your RTX 3050
# We use 224x224 to keep the VRAM usage low
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Test the loader
dataset = BrainDataset(
    csv_file='./data/prepared_2d_data/dataset_manifest.csv',
    root_dir='./data/prepared_2d_data/',
    transform=transforms
)

# Batch size of 16 is safe for 6GB VRAM with 2D images
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Loaded {len(dataset)} tumor slices.")
img, msk = next(iter(dataloader))
print(f"Batch Image Shape: {img.shape}") # Expect [16, 1, 224, 224]
print(f"Batch Mask Shape: {msk.shape}")  # Expect [16, 1, 224, 224]