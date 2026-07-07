"""
Training loop for the ResNet-50 transfer learning model.

- Optimizer: Adam
- LR schedule: cosine annealing (helps convergence vs. a fixed LR)
- Tracks train/val loss & accuracy per epoch -> learning curves
- Saves the best checkpoint (by validation accuracy) to disk
- Early stopping on validation accuracy plateau
"""

import os
import json
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE,
    CHECKPOINT_PATH, OUTPUT_DIR, EARLY_STOP_PATIENCE, FREEZE_BACKBONE,
)
from dataset import get_dataloaders
from model import build_model


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over `loader`. If optimizer is given, trains; else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(is_train)

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    torch.set_grad_enabled(True)
    return running_loss / total, correct / total


def train_model(optimizer_name="adam", lr_schedule="cosine",
                 num_epochs=NUM_EPOCHS, tag="baseline"):
    """
    Full training run. `optimizer_name` and `lr_schedule` are exposed as
    arguments (rather than hardcoded) so ablation.py can reuse this function
    to compare configurations.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    model = build_model(num_classes, freeze_backbone=FREEZE_BACKBONE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        scheduler = None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, epochs_no_improve = 0.0, 0
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{tag}] Epoch {epoch:02d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
            }, CHECKPOINT_PATH.replace(".pt", f"_{tag}.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{EARLY_STOP_PATIENCE} epochs).")
                break

    elapsed = time.time() - start
    print(f"Training finished in {elapsed/60:.1f} min. Best val acc: {best_val_acc:.4f}")

    with open(f"{OUTPUT_DIR}/history_{tag}.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_learning_curves(history, tag)
    return model, history, class_names, test_loader


def plot_learning_curves(history, tag="baseline"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/learning_curves_{tag}.png", dpi=150)
    plt.close()
    print(f"Saved learning curves to {OUTPUT_DIR}/learning_curves_{tag}.png")


if __name__ == "__main__":
    train_model()
