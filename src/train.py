"""
End-to-End Training and Evaluation Loop for Audio Event Classification.

Features:
- Hardware acceleration auto-detection (NVIDIA CUDA, Apple Silicon MPS, or CPU)
- Mixed-precision training support
- AdamW optimizer with Cosine Annealing learning rate schedule
- Gradient clipping to stabilize training
- Per-epoch training & validation metrics (Loss, Top-1 Accuracy)
- Comprehensive evaluation with classification report & confusion matrix
- Checkpoint saving for best validation performance
- Synthetic mock dataset fallback for zero-setup execution
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add parent directory to path to allow direct execution
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.dataset import AudioEventDataset, create_dataloaders
from src.dsp_pipeline import AudioConfig
from src.model import AudioSpectrogramCNN


def set_seed(seed: int = 42):
    """Sets random seeds across python, numpy, and pytorch for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Selects best available computing device: cuda, mps (Apple Silicon), or cpu."""
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """Runs a single training epoch across all batches.

    Returns
    -------
    Tuple[float, float]
        (average_epoch_loss, accuracy_percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for specs, labels in dataloader:
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)

        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        running_loss += loss.item() * specs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates model performance on validation / test split.

    Returns
    -------
    Tuple[float, float, np.ndarray, np.ndarray]
        (average_loss, accuracy_percentage, all_predictions, all_targets)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for specs, labels in dataloader:
            specs = specs.to(device)
            labels = labels.to(device)

            outputs = model(specs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(1, total)
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)


def train_model(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: Optional[str] = None,
    seed: int = 42,
    audio_config: Optional[AudioConfig] = None,
) -> Dict:
    """Executes full training pipeline, logging, and model checkpointing."""
    set_seed(seed)
    device = get_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_config = audio_config or AudioConfig()

    print(f"==================================================")
    print(f" Training Audio Event Classifier")
    print(f" Device:         {device}")
    print(f" Data directory: {data_dir}")
    print(f" Epochs:         {epochs} | Batch size: {batch_size} | LR: {learning_rate}")
    print(f"==================================================")

    # 1. Prepare DataLoaders
    train_loader, val_loader, class_to_idx, idx_to_class = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=0.8,
        config=audio_config,
        seed=seed,
        augment_train=True,
    )
    num_classes = len(class_to_idx)
    print(f"Discovered {num_classes} classes: {class_to_idx}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # 2. Instantiate Model
    model = AudioSpectrogramCNN(
        num_classes=num_classes,
        in_channels=1,
        base_channels=32,
        dropout=0.3,
    ).to(device)

    total_params, trainable_params = model.count_parameters()
    print(f"Model parameters: {trainable_params:,} trainable ({total_params:,} total)")

    # 3. Setup Optimization & Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc, val_preds, val_targets = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_duration = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "audio_config": audio_config.__dict__,
                },
                checkpoint_path,
            )

        star = " *" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | "
            f"LR: {current_lr:.1e} | Time: {epoch_duration:.1f}s{star}"
        )

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.2f}s. Best Validation Accuracy: {best_val_acc:.2f}%")

    # 5. Final Evaluation on Best Checkpoint
    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

    class_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(
        val_targets, val_preds, target_names=class_names, digits=4, output_dict=True
    )
    print("\nFinal Validation Classification Report:")
    print(classification_report(val_targets, val_preds, target_names=class_names, digits=4))

    conf_mat = confusion_matrix(val_targets, val_preds).tolist()

    # Save metrics and history
    results = {
        "best_val_acc": best_val_acc,
        "history": history,
        "classification_report": report,
        "confusion_matrix": conf_mat,
        "class_to_idx": class_to_idx,
    }
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Audio Event Classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to audio dataset directory with class subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints and training logs.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', 'mps').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_path = Path(args.data_dir) if args.data_dir else None

    # If data_dir not provided or empty, generate mock data automatically
    if data_path is None or not data_path.exists():
        print("No valid dataset directory specified. Generating synthetic mock dataset...")
        from tests.generate_mock import generate_synthetic_dataset

        mock_dir = parent_dir / "data" / "mock_dataset"
        generate_synthetic_dataset(output_dir=mock_dir, num_samples_per_class=15)
        data_path = mock_dir

    output_path = Path(args.output_dir)
    train_model(
        data_dir=data_path,
        output_dir=output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
