import os
import json
import itertools
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
from joblib import Parallel, delayed
from utils.model import TumorCNN
from utils.dataset import get_dataloaders
from utils.training_utils import (
    get_device, get_optimizer_and_scheduler, setup_amp,
    train_one_epoch, evaluate
)

def run_trial(config, trial_dir, trial_num=None, total_trials=None):
    """Run a single training trial with given config."""
    device = get_device(config["device"], verbose=False)
    config["device"] = device

    os.makedirs(trial_dir, exist_ok=True)
    train_loader, test_loader, classes = get_dataloaders(config)

    model = TumorCNN(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    use_amp, scaler = setup_amp(device)

    best_acc = 0.0
    best_model_path = os.path.join(trial_dir, "best_model.pth")
    epochs = config["training"]["epochs"]
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    trial_desc = f"Trial {trial_num}/{total_trials}" if trial_num and total_trials else "Trial"
    epoch_pbar = tqdm(range(1, epochs + 1), desc=trial_desc, unit="epoch", leave=False)

    for epoch in epoch_pbar:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler,
            epoch=epoch, total_epochs=epochs
        )
        val_loss, val_acc, _, _ = evaluate(
            model, test_loader, criterion, device, use_amp,
            epoch=epoch, total_epochs=epochs, stage="Val"
        )
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        epoch_pbar.set_postfix_str(f"Loss: {train_loss:.4f}→{val_loss:.4f} | Acc: {train_acc:.3f}→{val_acc:.3f}")

    result = {"config": config, "best_val_acc": float(best_acc), "history": history}
    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return float(best_acc)

def execute_single_combo(i, combo, keys, base_config, results_dir, total_trials=None):
    """Helper function to run a trial inside the parallel loop."""
    # Build config for this trial
    config = json.loads(json.dumps(base_config))
    for key, value in zip(keys, combo):
        if key in config["training"]:
            config["training"][key] = value

    lr, bs, dropout = combo
    trial_dir = os.path.join(results_dir, f"trial_{i:02d}_lr{lr:.0e}_bs{bs}_do{dropout:.1f}")

    try:
        val_acc = run_trial(config, trial_dir, trial_num=i, total_trials=total_trials)
        return {"trial": i, "learning_rate": lr, "batch_size": bs, "dropout_rate": dropout, "best_val_acc": val_acc}
    except Exception as e:
        print(f"Trial {i} failed: {e}")
        return None

def grid_search(base_config_path="config.json"):
    with open(base_config_path) as f:
        base_config = json.load(f)

    param_grid = {
        "learning_rate": [5e-4, 1e-3, 2e-3],
        "batch_size": [64, 128],
        "dropout_rate": [0.3, 0.4, 0.5],
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"grid_search_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    total_trials = len(combinations)
    print(f"\n{'='*70}")
    print(f"Grid Search: {total_trials} trials")
    print(f"Parameters: LR={param_grid['learning_rate']}, BS={param_grid['batch_size']}, Dropout={param_grid['dropout_rate']}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*70}\n")

    start_time = time.time()

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(execute_single_combo)(i, combo, keys, base_config, results_dir, total_trials=total_trials)
        for i, combo in enumerate(combinations, 1)
    )

    results = [r for r in results if r is not None]

    elapsed = time.time() - start_time
    summary = {
        "total_trials": total_trials,
        "completed_trials": len(results),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
        "results": sorted(results, key=lambda x: x["best_val_acc"], reverse=True)
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Grid Search Complete!")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Completed trials: {len(results)}/{total_trials}")
    if results:
        best = results[0]
        print(f"\nBest configuration:")
        print(f"  LR: {best['learning_rate']}, Batch Size: {best['batch_size']}, Dropout: {best['dropout_rate']}")
        print(f"  Val Accuracy: {best['best_val_acc']:.4f}")
    print(f"{'='*70}\n")

    return results_dir, summary

if __name__ == "__main__":
    grid_search()