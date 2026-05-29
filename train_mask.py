import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # Professional progress bar
import os

# Verbatim references to your project files
from utils.mask_model import UNet, device, get_model, model 
from utils.braindataset import BrainDataset, transforms 


# 1. Hyperparameters
EPOCHS = 30
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4

def train():
    device = torch.device("mps")
    print(f"🚀 Training starting on {device}")
    # 2. Setup Data with M1 Pro Optimizations
    full_dataset = BrainDataset(
        csv_file='./data/prepared_2d_data/dataset_manifest.csv',
        root_dir='./data/prepared_2d_data/',
        transform=transforms
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Optimized DataLoaders for Apple Silicon
    # num_workers=8 utilizes the performance cores of the M1 Pro
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=8,        # Use multiple CPU cores for preprocessing
        pin_memory=False,      # Faster transfer to GPU
        persistent_workers=True 
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4
    )

    # 3. Loss & Optimizer
    def dice_loss(pred, target, smooth=1.):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Enhanced Training Loop
    print(f"🚀 Training starting on {device} (M1 Pro Accelerated)")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Progress bar setup
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward
            outputs = model(images)
            loss = torch.nn.functional.binary_cross_entropy(outputs, masks) + dice_loss(outputs, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Update progress bar with live loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Validation
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for v_imgs, v_msks in val_loader:
                v_imgs, v_msks = v_imgs.to(device), v_msks.to(device)
                v_outs = model(v_imgs)
                val_dice += (1 - dice_loss(v_outs, v_msks)).item()

        avg_val_dice = val_dice / len(val_loader)
        print(f"✅ Epoch {epoch+1} Complete | Avg Loss: {train_loss/len(train_loader):.4f} | Val Dice Score: {avg_val_dice:.4f}")

        # Checkpoint: Save if it's the best model yet
        if avg_val_dice > 0.8:
            torch.save(model.state_dict(), f"best_brain_model_dice_{avg_val_dice:.2f}.pth")

    torch.save(model.state_dict(), "final_brain_tumor_unet.pth")

if __name__ == '__main__':
    model, device = get_model()
    train()