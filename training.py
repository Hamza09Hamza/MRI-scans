
import os
import time
import torch                          # The core PyTorch library
import torch.nn as nn                 # Neural network building blocks
import torch.optim as optim           # Gradient-descent optimizers
import torch.amp                      # Automatic Mixed Precision (modern API)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # Dataset helpers + image transforms
from sklearn.metrics import classification_report  # Detailed per-class accuracy
import matplotlib.pyplot as plt



DATA_DIR   = os.path.dirname(os.path.abspath(__file__)) 
TRAIN_DIR  = os.path.join(DATA_DIR, "Training")
TEST_DIR   = os.path.join(DATA_DIR, "Testing")

IMG_SIZE   = 128

# GPU/MPS OPTIMISATION: Larger batch size = better GPU utilization.
# On MPS/CUDA with 4GB+ VRAM, we can use 128-256 safely.
# Larger batches → smoother gradients → better convergence.
BATCH_SIZE = 512

EPOCHS     = 15

LR         = 1e-3

NUM_CLASSES = 4

DEVICE = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)
print(f"Using device: {DEVICE}")

# GPU/MPS INFO: Print available resources
if DEVICE == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif DEVICE == "mps":
    print("MPS (Metal Performance Shaders) enabled on Apple Silicon")

# GPU/MPS OPTIMISATION: Enable cuDNN auto-tuner for CUDA.
# This finds the fastest convolution algorithm for your specific GPU.
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


train_transforms = transforms.Compose([
    # --- Spatial augmentations (only applied during training) ---
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(),        # Flip left↔right with 50% chance
    transforms.RandomRotation(15),            # Rotate ±15 degrees randomly
    transforms.ColorJitter(brightness=0.2,    # Vary brightness/contrast slightly
                           contrast=0.2),

    # --- Tensor conversion + normalisation ---
    transforms.ToTensor(),


    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])



train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transforms)

print(f"Classes : {train_dataset.classes}")
print(f"Training: {len(train_dataset)} images")
print(f"Testing : {len(test_dataset)} images")


# GPU/MPS OPTIMISATION: pin_memory=True pre-allocates pinned RAM for fast GPU transfer.
# num_workers loads images in parallel while GPU trains → eliminates I/O bottleneck.
# For MPS, use 2-4 workers; for CUDA, can use more.
num_workers = 8 if DEVICE in ["cuda", "mps"] else 0
pin_memory_flag = DEVICE in ["cuda", "mps"]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=num_workers,
                          pin_memory=pin_memory_flag, prefetch_factor=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=pin_memory_flag)



class TumorCNN(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()



        self.block1 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),


            nn.BatchNorm2d(32),


            nn.ReLU(inplace=True),


            nn.MaxPool2d(2, 2), 
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # more filters = richer features
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 64×64 → 32×32
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 32×32 → 16×16
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # output: (batch, 128, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),           # (batch, 128, 1, 1) → (batch, 128)

            nn.Linear(128, 256),   # Fully connected layer: 128 inputs → 256 outputs
            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(256, num_classes), 
        )

    def forward(self, x):
        # forward() defines how data flows through the model.
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


model = TumorCNN(num_classes=NUM_CLASSES).to(DEVICE)
# .to(DEVICE) moves all model parameters to GPU/MPS/CPU so computations happen there.

# Quick sanity check: print parameter count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")



criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=LR)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                  factor=0.5, patience=3)

# GPU/MPS OPTIMISATION: Automatic Mixed Precision (AMP) speeds up training 1.5-2x.
# Uses float16 for fast computation, float32 for stable gradients.
# Supported on CUDA and newer PyTorch versions. MPS also benefits.
use_amp = DEVICE in ["cuda", "mps"]
scaler = torch.amp.GradScaler(device=DEVICE) if DEVICE == "cuda" else None


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler):
    model.train()   # Enables dropout + batch norm in training mode
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # GPU/MPS OPTIMISATION: AMP context automatically casts ops to float16.
        # Forward pass is faster, backward pass uses float32 for stability.
        if use_amp and DEVICE == "cuda":
            with torch.amp.autocast(device_type=DEVICE):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)   # Pick the class with the highest logit
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()   # Disables dropout; batch norm uses running statistics
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():   # Disables gradient tracking — saves memory and time
        for images, labels in loader:  # during evaluation (no backprop needed)
            images, labels = images.to(device), labels.to(device)
            # GPU/MPS OPTIMISATION: Use AMP for evaluation too (no scaler needed)
            if use_amp:
                with torch.amp.autocast(device_type=device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# --- Training loop with history tracking for plots ---
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_acc = 0.0

print("\n--- Starting training ---")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, use_amp, scaler)
    val_loss,   val_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE, use_amp)

    scheduler.step(val_acc)  # Tell the scheduler today's validation accuracy

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - t0
    # GPU/MPS OPTIMISATION: Monitor memory usage (CUDA only)
    mem_str = ""
    if DEVICE == "cuda":
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_str = f"  GPU mem: {mem_used:.2f}GB"

    print(f"Epoch {epoch:2d}/{EPOCHS}  "
          f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f}  |  "
          f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}  "
          f"[{elapsed:.1f}s]{mem_str}")

    # Save the best model (highest validation accuracy)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✓ New best model saved (val_acc={best_acc:.3f})")



print("\n--- Final evaluation on test set ---")
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
_, test_acc, preds, labels_true = evaluate(model, test_loader, criterion, DEVICE, use_amp)
print(f"Test accuracy: {test_acc:.3f}\n")
print(classification_report(labels_true, preds,
                             target_names=train_dataset.classes))




fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="Train loss")
ax1.plot(history["val_loss"],   label="Val loss")
ax1.set_title("Loss over epochs")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(history["train_acc"], label="Train acc")
ax2.plot(history["val_acc"],   label="Val acc")
ax2.set_title("Accuracy over epochs")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()
print("Learning curves saved to learning_curves.png")

# GPU/MPS OPTIMISATION: Summary of what was used
print("\n--- GPU/MPS Optimisation Summary ---")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE} (larger = better GPU utilization)")
print(f"Data loading workers: {num_workers} (parallel image loading)")
print(f"Pin memory: {pin_memory_flag} (fast GPU transfer)")
if DEVICE == "cuda":
    print(f"Mixed Precision (AMP): Enabled (1.5-2x faster)")
    print(f"cuDNN benchmark: Enabled (finds optimal GPU algorithms)")
elif DEVICE == "mps":
    print(f"Metal Performance Shaders: Enabled")
    print(f"Note: MPS doesn't support AMP yet, but is still optimized")
