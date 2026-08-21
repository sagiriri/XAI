"""
One-off script to peek inside a CIFAR-10 batch file — these are Python
pickle files, not something you can double-click open. Run with
xaibench-env active.
"""

import pickle
import numpy as np
from PIL import Image

BATCH_PATH = r"D:\XAI\data\cifar-10-batches-py\data_batch_1"

with open(BATCH_PATH, "rb") as f:
    batch = pickle.load(f, encoding="bytes")

print("Keys in this file:", list(batch.keys()))
print("Number of images:", len(batch[b"data"]))
print("Labels (first 20):", batch[b"labels"][:20])

# Each image is stored as a flat 3072-length array (32x32x3), channel-first.
# Reshape it back into a normal viewable image and save one as PNG.
img_flat = batch[b"data"][0]
img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)  # -> (32, 32, 3)
Image.fromarray(img).save("cifar_sample.png")
print("\nSaved first image as cifar_sample.png — open that to actually see it.")

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
print(f"This image's label: {class_names[batch[b'labels'][0]]}")
