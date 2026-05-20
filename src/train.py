"""
Training script for Dog Breed Identification.

This script:
1. Loads the dataset via create_dataloaders()
2. Initializes the model (baseline CNN)
3. Runs the training loop for N epochs
4. Logs all metrics to Weights & Biases (wandb)
5. Saves the best model checkpoint

Usage:
    python -m src.train
"""

import torch
import torch.nn as nn
import wandb
from pathlib import Path
from tqdm import tqdm

from src.dataloaders.dataset import create_dataloaders
from src.models.baseline import Baseline


# ─── Configuration ──────────────────────────────────────────────────────────
# All hyperparameters in one place for clarity and wandb tracking.

CONFIG = {
    "model": "baseline_cnn",
    "num_classes": 120,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "val_split": 0.2,
    "seed": 42,
}


def get_device():
    """Select best available device: MPS (Apple Silicon), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Returns:
        avg_loss: mean loss over all batches
        accuracy: fraction of correct predictions
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set.

    No gradients computed — purely forward passes for evaluation.

    Returns:
        avg_loss: mean loss over all batches
        accuracy: fraction of correct predictions
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc="  Val", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # ─── Setup ──────────────────────────────────────────────────────────
    device = get_device()
    print(f"Device: {device}")

    # Initialize wandb — creates a run on the wandb dashboard
    wandb.init(
        project="dog-breed-identification",
        config=CONFIG,
        tags=["baseline", "phase2"],
    )

    # ─── Data ───────────────────────────────────────────────────────────
    train_loader, val_loader, breed_to_idx, idx_to_breed = create_dataloaders(
        data_dir="data/",
        batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        val_split=CONFIG["val_split"],
        num_workers=4,
        seed=CONFIG["seed"],
    )

    # ─── Model ──────────────────────────────────────────────────────────
    model = Baseline(num_classes=CONFIG["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Tell wandb to track gradients and parameters (useful for debugging)
    wandb.watch(model, criterion, log="all", log_freq=50)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ─── Training loop ──────────────────────────────────────────────────
    best_val_acc = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        })

        print(f"  Train — loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / "best_baseline.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "breed_to_idx": breed_to_idx,
            }, checkpoint_path)
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    # ─── Final summary ──────────────────────────────────────────────────
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    wandb.log({"best_val_accuracy": best_val_acc})
    wandb.finish()


if __name__ == "__main__":
    main()
