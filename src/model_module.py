"""
XAIBench — Model Module

Provides a unified interface for three model families:
  1. SimpleCNN   -> lightweight custom baseline, trained from scratch
  2. ResNet      -> torchvision's resnet18 / resnet50, optionally pretrained
  3. EfficientNet -> torchvision's efficientnet_b0, optionally pretrained

All three expose the same interface (get_model, train_model, evaluate_model)
so the Benchmark Engine can treat them interchangeably later.

Usage:
    from model_module import get_model, train_model, evaluate_model

    model = get_model("resnet18", num_classes=10, pretrained=True)
    model = train_model(model, train_loader, test_loader, epochs=5, device="cpu")
    metrics = evaluate_model(model, test_loader, device="cpu")
"""

import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models


# ---------------------------------------------------------------------------
# 1. Custom baseline CNN
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    A small 4-block CNN. Not meant to compete with ResNet/EfficientNet on
    accuracy — its purpose is to serve as a fast, fully-understood baseline
    so the Benchmark Engine has a lower-complexity reference point, and so
    XAI methods have a simpler model to explain when first testing a new
    metric.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 28 -> 1x1, any input size works
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 2 & 3. ResNet / EfficientNet via torchvision, with the final layer swapped
#         to match num_classes
# ---------------------------------------------------------------------------

def _build_resnet(variant: str, num_classes: int, pretrained: bool):
    weights = None
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT if variant == "resnet18" else models.ResNet50_Weights.DEFAULT

    if variant == "resnet18":
        model = models.resnet18(weights=weights)
    elif variant == "resnet50":
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown resnet variant '{variant}'")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_efficientnet(num_classes: int, pretrained: bool):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_model(model_name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Factory function — the single entry point the rest of XAIBench should use.

    Args:
        model_name: one of "simplecnn", "resnet18", "resnet50", "efficientnet"
        num_classes: number of output classes (10 for CIFAR-10, 50 for FunnyBirds, etc.)
        pretrained: if True, loads ImageNet-pretrained weights for ResNet/EfficientNet
                    (ignored for SimpleCNN, which always trains from scratch).
                    Set this False if you want a fair "trained from scratch" comparison
                    against SimpleCNN, or True for faster convergence / higher accuracy.

    Returns:
        An nn.Module ready for training or inference.
    """
    model_name = model_name.lower()

    if model_name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return _build_resnet("resnet18", num_classes, pretrained)
    elif model_name == "resnet50":
        return _build_resnet("resnet50", num_classes, pretrained)
    elif model_name == "efficientnet":
        return _build_efficientnet(num_classes, pretrained)
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Use one of: simplecnn, resnet18, resnet50, efficientnet"
        )


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader,
    test_loader=None,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """
    Trains `model` in place and returns it.

    Args:
        test_loader: if provided, prints test accuracy after each epoch
                     (useful to watch for overfitting, but roughly doubles
                     time per epoch — pass None to skip and just train faster).
        lr: 1e-4 is a safe default for fine-tuning pretrained ResNet/EfficientNet.
            If training SimpleCNN from scratch, a higher lr like 1e-3 usually
            converges faster.
    """
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
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

        train_loss = running_loss / total
        train_acc = correct / total
        elapsed = time.time() - start_time

        if verbose:
            msg = f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | time={elapsed:.1f}s"
            if test_loader is not None:
                test_metrics = evaluate_model(model, test_loader, device=device)
                msg += f" | test_acc={test_metrics['accuracy']:.4f}"
                model.train()  # evaluate_model() sets eval mode; switch back for next epoch
            print(msg)

    return model


def evaluate_model(model: nn.Module, test_loader, device: str = "cpu") -> dict:
    """
    Runs inference over test_loader and returns accuracy + average per-batch
    inference time. Precision/Recall/F1/AUC are computed separately in the
    Benchmark Engine (which has access to scikit-learn and can handle
    multi-class averaging strategies) — this function focuses on the two
    metrics that are cheap to compute inline during any evaluation pass.
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    total_inference_time = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start = time.time()
            outputs = model(images)
            total_inference_time += time.time() - start
            num_batches += 1

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_batch_inference_time = total_inference_time / num_batches

    return {
        "accuracy": accuracy,
        "avg_batch_inference_time_sec": avg_batch_inference_time,
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    Returns the model's parameter storage size in MB. Used by the Benchmark
    Engine's "Model Size" metric — one of the accuracy/speed/size trade-offs
    the whole XAIBench comparison is built around.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(model_name: str, path: str, num_classes: int = 10, device: str = "cpu") -> nn.Module:
    """
    Rebuilds the architecture (with pretrained=False, since we're about to
    overwrite the weights anyway) and loads saved weights into it.
    """
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Quick manual test — run this file directly to sanity-check all three models
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from dataset_module import get_cifar_loaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading a small CIFAR-10 subset for a quick smoke test...")
    train_loader, test_loader, class_names = get_cifar_loaders(
        dataset_name="cifar10", data_dir="../data", batch_size=16, image_size=224, num_workers=0
    )

    # Shrink to a tiny subset so this smoke test runs in under a minute,
    # instead of a full epoch over all 50,000 images.
    from torch.utils.data import Subset, DataLoader
    small_train = DataLoader(Subset(train_loader.dataset, range(64)), batch_size=16, shuffle=True)
    small_test = DataLoader(Subset(test_loader.dataset, range(32)), batch_size=16, shuffle=False)

    for model_name in ["simplecnn", "resnet18", "efficientnet"]:
        print("\n" + "=" * 50)
        print(f"Testing: {model_name}")
        print("=" * 50)

        model = get_model(model_name, num_classes=len(class_names), pretrained=(model_name != "simplecnn"))
        print(f"Model size: {get_model_size_mb(model):.2f} MB")

        model = train_model(model, small_train, test_loader=small_test, epochs=1, device=device)

        metrics = evaluate_model(model, small_test, device=device)
        print(f"Final metrics: {metrics}")

    print("\nModel Module sanity check complete.")
