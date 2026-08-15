"""
XAIBench — Dataset Module

Handles loading and preprocessing for two dataset types:
  1. CIFAR-10 / CIFAR-100  -> used to train/test the Model Module (CNN, ResNet, EfficientNet)
  2. FunnyBirds             -> synthetic dataset with ground-truth part importances,
                                used later by the XAI Module to evaluate explanation faithfulness

Usage:
    from dataset_module import get_cifar_loaders, get_funnybirds_loaders

    train_loader, test_loader, class_names = get_cifar_loaders(
        dataset_name="cifar10", data_dir="../data", batch_size=64
    )
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


# ---------------------------------------------------------------------------
# CIFAR-10 / CIFAR-100
# ---------------------------------------------------------------------------

# Standard normalization stats for CIFAR (computed over the training set).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_cifar_transforms(train: bool, image_size: int = 224):
    """
    Returns the preprocessing pipeline for CIFAR images.

    image_size=224 is used (instead of CIFAR's native 32x32) because
    ResNet/EfficientNet architectures expect larger inputs. If you're
    training a small custom CNN from scratch, you can pass image_size=32
    instead to skip the upscaling and train faster.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=int(image_size * 0.125)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])


def get_cifar_loaders(
    dataset_name: str = "cifar10",
    data_dir: str = "../data",
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 2,
):
    """
    Downloads (if needed) and loads CIFAR-10 or CIFAR-100.

    Args:
        dataset_name: "cifar10" or "cifar100"
        data_dir: folder to store/read the dataset from
        batch_size: batch size for both loaders
        image_size: images are resized to (image_size, image_size)
        num_workers: dataloader worker processes (set to 0 on Windows if you
                      hit multiprocessing errors)

    Returns:
        train_loader, test_loader, class_names (list of str)
    """
    dataset_name = dataset_name.lower()
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name == "cifar10":
        dataset_cls = datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset_cls = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Use 'cifar10' or 'cifar100'.")

    train_set = dataset_cls(
        root=data_dir, train=True, download=True,
        transform=get_cifar_transforms(train=True, image_size=image_size),
    )
    test_set = dataset_cls(
        root=data_dir, train=False, download=True,
        transform=get_cifar_transforms(train=False, image_size=image_size),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_set.classes
    return train_loader, test_loader, class_names


# ---------------------------------------------------------------------------
# FunnyBirds
# ---------------------------------------------------------------------------

class FunnyBirdsDataset(Dataset):
    """
    Loads the FunnyBirds dataset for classification.

    Expects the folder structure produced by unzipping FunnyBirds.zip:
        FunnyBirds/
          train/
            <class_id>/
              <image>.png
          test/
            <class_id>/
              <image>.png

    Note: this class handles the classification images only. FunnyBirds also
    ships part-level intervention tools (for measuring ground-truth part
    importance) under its own scripts in the funnybirds-framework repo —
    those get wired in separately inside the XAI Module when we build the
    faithfulness metrics, not here in the Dataset Module.
    """

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 224,
                 return_paths: bool = False):
        self.root_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"Expected FunnyBirds folder at '{self.root_dir}'. "
                f"Make sure you've unzipped FunnyBirds.zip into '{root_dir}'."
            )

        # return_paths=True also hands back each sample's matching
        # <split>_part_map/ path — the ground-truth segmentation mask used
        # by funnybirds_ground_truth.py to check whether an explanation
        # actually points at a real bird part vs. background clutter.
        # Off by default so existing callers (training scripts, anything
        # expecting a plain (image, label) tuple) are unaffected.
        self.return_paths = return_paths
        self.part_map_root_dir = os.path.join(root_dir, f"{split}_part_map")

        # Sorted numerically, not alphabetically: folder names are '0'..'49',
        # and plain sorted() would order them as strings ('0','1','10','11'...),
        # which mismatches classes.json's numeric class_idx and would silently
        # scramble labels during training.
        self.classes = sorted(os.listdir(self.root_dir), key=lambda x: int(x))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, fname)
                    # Ground-truth part map lives at the same class/filename,
                    # just under <split>_part_map/ instead of <split>/.
                    part_map_path = os.path.join(self.part_map_root_dir, class_name, fname)
                    self.samples.append((img_path, self.class_to_idx[class_name], part_map_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under '{self.root_dir}'. Check the dataset was unzipped correctly.")

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, part_map_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if self.return_paths:
            return image, label, part_map_path
        return image, label


def get_funnybirds_loaders(
    data_dir: str = "../data/FunnyBirds",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    return_paths: bool = False,
):
    """
    Loads FunnyBirds train/test sets.

    Args:
        return_paths: if True, each batch also yields the matching
                       train_part_map/test_part_map path per sample —
                       needed for funnybirds_ground_truth.py's part-overlap
                       metric. Leave False for training, where you just
                       want (images, labels).

    Returns:
        train_loader, test_loader, class_names (list of str)
    """
    train_set = FunnyBirdsDataset(data_dir, split="train", image_size=image_size, return_paths=return_paths)
    test_set = FunnyBirdsDataset(data_dir, split="test", image_size=image_size, return_paths=return_paths)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_set.classes


# ---------------------------------------------------------------------------
# Quick manual test — run this file directly to sanity-check both loaders
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("Testing CIFAR-10 loader")
    print("=" * 50)
    train_loader, test_loader, classes = get_cifar_loaders(
        dataset_name="cifar10", data_dir="../data", batch_size=8, image_size=224, num_workers=0
    )
    images, labels = next(iter(train_loader))
    print(f"Classes: {classes}")
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"Train set size: {len(train_loader.dataset)} | Test set size: {len(test_loader.dataset)}")

    print("\n" + "=" * 50)
    print("Testing FunnyBirds loader (skipped if not downloaded yet)")
    print("=" * 50)
    try:
        fb_train_loader, fb_test_loader, fb_classes = get_funnybirds_loaders(
            data_dir="../data/FunnyBirds", batch_size=8, image_size=224, num_workers=0
        )
        images, labels = next(iter(fb_train_loader))
        print(f"Classes: {fb_classes}")
        print(f"Batch image shape: {images.shape}")
        print(f"Train set size: {len(fb_train_loader.dataset)} | Test set size: {len(fb_test_loader.dataset)}")
    except FileNotFoundError as e:
        print(f"Skipped: {e}")

    print("\nDataset Module sanity check complete.")
