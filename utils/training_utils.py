import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp
from tqdm import tqdm


def get_device(config_device="auto", verbose=True):
    """Detect and return the best available device."""
    if config_device == "auto":
        device = (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
    else:
        device = config_device

    if verbose:
        print(f"\n{'='*60}")
        print(f"Using device: {device}")

        if device == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            torch.backends.cudnn.benchmark = True
            print("✓ cuDNN auto-tuner enabled")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    return device


def get_optimizer_and_scheduler(model, config):
    """Create optimizer and learning rate scheduler."""
    opt_config = config["optimizer"]
    lr = config["training"]["learning_rate"]

    if opt_config["type"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=opt_config["betas"],
            weight_decay=opt_config["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['type']}")

    sched_config = config["scheduler"]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_config["mode"],
        factor=sched_config["factor"],
        patience=sched_config["patience"]
    )

    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler, epoch=None, total_epochs=None):
    """
    Optimized training loop for M1 Pro (MPS) and CUDA.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    desc = "Train"
    if epoch is not None and total_epochs is not None:
        desc = f"Epoch {epoch}/{total_epochs} [Train]"

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        # Move data to the M1 GPU/MPS context[cite: 5]
        images, labels = images.to(device), labels.to(device)

        # set_to_none=True is more efficient on Apple Silicon than zeroing out
        optimizer.zero_grad(set_to_none=True)

        # Use the generic device_type to enable AMP on MPS (requires PyTorch 2.3+)
        if use_amp:
            with torch.amp.autocast(device_type=device):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # If setup_amp provided a scaler (supported on recent MPS versions)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard backward if scaler isn't used for MPS
                loss.backward()
                optimizer.step()
        else:
            # Full precision path[cite: 5, 8]
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Metrics calculation
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Update progress bar with current metrics
        current_loss = running_loss / total
        current_acc = correct / total
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.3f}"})

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device, use_amp, epoch=None, total_epochs=None, stage="Val"):
    """Evaluate model on dataset."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    desc = stage
    if epoch is not None and total_epochs is not None:
        desc = f"Epoch {epoch}/{total_epochs} [{stage}]"

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

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
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar with current metrics
            current_loss = running_loss / total
            current_acc = correct / total
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.3f}"})

    return running_loss / total, correct / total, all_preds, all_labels


def setup_amp(device):
    """Enable GradScaler for both CUDA and MPS."""
    use_amp = device in ["cuda", "mps"]
    
    # Newer PyTorch versions support GradScaler on mps device
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler(device=device)
        except Exception:
            # Fallback if your PyTorch version is slightly older
            scaler = torch.amp.GradScaler(device="cuda") if device == "cuda" else None
            
    return use_amp, scaler
# In train_one_epoch
