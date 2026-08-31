"""
IRIS-XAI — Full Training Script

Trains all three models (SimpleCNN, ResNet18, EfficientNet) on the full
CIFAR-10 and FunnyBirds datasets, with:

  - Per-epoch history logged to CSV (for a learning-curve plot in your report)
  - Checkpoints saved after every model, so an interrupted run can be resumed
    without retraining everything from scratch
  - Mixed precision (torch.cuda.amp) to fit comfortably in 4GB VRAM
  - Skip-if-already-trained logic (pass --force to retrain anyway)

Usage (from src/, with iris-xai-env active):
    python train_full.py
    python train_full.py --datasets cifar10                 # just one dataset
    python train_full.py --models resnet18 efficientnet     # just some models
    python train_full.py --epochs 15                        # override epoch count
    python train_full.py --force                            # retrain everything, ignore checkpoints

Output:
    ../models/<dataset>_<model_name>.pth              — trained weights
    ../results/training_history_<dataset>_<model>.csv — per-epoch loss/acc/time
"""

import os
import time
import argparse
import csv

import torch
import torch.nn as nn
from torch.optim import Adam

from dataset_module import get_cifar_loaders, get_funnybirds_loaders
from model_module import get_model, evaluate_model, save_model, get_model_size_mb


MODELS_DIR = "../models"
RESULTS_DIR = "../results"

# Batch sizes tuned for a 4GB VRAM laptop GPU at 224x224. SimpleCNN is cheap
# enough to go higher; ResNet/EfficientNet need to stay modest to avoid
# CUDA out-of-memory. If you hit an OOM error anyway, halve these.
BATCH_SIZES = {
    "simplecnn": 64,
    "resnet18": 24,
    "resnet50": 16,
    "efficientnet": 24,
}

NUM_CLASSES = {
    "cifar10": 10,
    "funnybirds": 50,
}


def get_loaders(dataset_name, batch_size):
    if dataset_name == "cifar10":
        return get_cifar_loaders(
            dataset_name="cifar10", data_dir="../data",
            batch_size=batch_size, image_size=224, num_workers=0,
        )
    elif dataset_name == "funnybirds":
        train_loader, test_loader, classes = get_funnybirds_loaders(
            data_dir="../data/FunnyBirds", batch_size=batch_size,
            image_size=224, num_workers=0,
        )
        return train_loader, test_loader, classes
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")


def train_one_model(model_name, dataset_name, epochs, device, use_amp, force):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    checkpoint_path = os.path.join(MODELS_DIR, f"{dataset_name}_{model_name}.pth")
    history_path = os.path.join(RESULTS_DIR, f"training_history_{dataset_name}_{model_name}.csv")

    if os.path.exists(checkpoint_path) and not force:
        print(f"[skip] {checkpoint_path} already exists (pass --force to retrain)")
        return

    print("\n" + "=" * 60)
    print(f"Training {model_name} on {dataset_name} for {epochs} epochs")
    print("=" * 60)

    batch_size = BATCH_SIZES.get(model_name, 16)
    num_classes = NUM_CLASSES[dataset_name]

    train_loader, test_loader, classes = get_loaders(dataset_name, batch_size)
    print(f"Train set: {len(train_loader.dataset)} images | Test set: {len(test_loader.dataset)} images | batch_size={batch_size}")

    model = get_model(model_name, num_classes=num_classes, pretrained=(model_name != "simplecnn"))
    model.to(device)
    print(f"Model size: {get_model_size_mb(model):.2f} MB")

    lr = 1e-3 if model_name == "simplecnn" else 1e-4
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []

    try:
        for epoch in range(epochs):
            model.train()
            start_time = time.time()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total
            elapsed = time.time() - start_time

            test_metrics = evaluate_model(model, test_loader, device=device)
            test_acc = test_metrics["accuracy"]

            print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | "
                  f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | time={elapsed:.1f}s")

            history.append({
                "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                "test_acc": test_acc, "epoch_time_sec": elapsed,
            })

            # Save history after every epoch, not just at the end — so a crash
            # or interruption mid-run still leaves you a usable partial curve.
            with open(history_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[CUDA OUT OF MEMORY] on {model_name}/{dataset_name}.")
            print(f"Try lowering BATCH_SIZES['{model_name}'] in this script (currently {batch_size}) and rerun.")
            torch.cuda.empty_cache()
            return
        else:
            raise

    save_model(model, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "funnybirds"],
                         choices=["cifar10", "funnybirds"])
    parser.add_argument("--models", nargs="+", default=["simplecnn", "resnet18", "efficientnet"],
                         choices=["simplecnn", "resnet18", "resnet50", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--force", action="store_true", help="retrain even if a checkpoint already exists")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: CUDA not available, falling back to CPU. This will be very slow for 22 epochs.")

    use_amp = (device == "cuda") and not args.no_amp

    total_runs = len(args.datasets) * len(args.models)
    run_num = 0
    start_all = time.time()

    for dataset_name in args.datasets:
        for model_name in args.models:
            run_num += 1
            print(f"\n[{run_num}/{total_runs}]", end=" ")
            train_one_model(model_name, dataset_name, args.epochs, device, use_amp, args.force)

    total_elapsed = time.time() - start_all
    print(f"\nAll training complete in {total_elapsed / 60:.1f} minutes.")
    print(f"Checkpoints in {MODELS_DIR}/, per-epoch histories in {RESULTS_DIR}/")
    print("Next: re-run benchmark_module.py pointing at these checkpoints to get real (non-smoke-test) results.")


if __name__ == "__main__":
    main()
