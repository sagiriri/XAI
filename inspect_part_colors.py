"""
One-off inspection script — NOT part of the pipeline, just for figuring out
the exact canonical color palette FunnyBirds uses in train_part_map/.

Run from anywhere (doesn't need xaibench-env activated, just PIL):
    python inspect_part_colors.py

Prints every distinct color in the image along with how many pixels use it,
sorted largest first. The background (pure black, (0,0,0)) and the
gray body/clutter blobs will dominate — what we actually care about are the
smaller, saturated colors (yellow/blue/green/red/etc.) since those are the
diagnostic parts (beak/eye/foot/tail/wing).
"""

from PIL import Image
from collections import Counter

# Change this path if needed — defaults to the exact image you showed me.
IMAGE_PATH = r"D:\XAI\data\FunnyBirds\train_part_map\3\003000.png"

img = Image.open(IMAGE_PATH).convert("RGB")
pixels = list(img.getdata())
counts = Counter(pixels)

print(f"Image size: {img.size}, total pixels: {len(pixels)}")
print(f"Number of distinct colors: {len(counts)}\n")
print(f"{'RGB':<18} {'Pixel count':<12} {'% of image':<10}")
print("-" * 42)
for color, count in counts.most_common(25):
    pct = 100 * count / len(pixels)
    print(f"{str(color):<18} {count:<12} {pct:.2f}%")

# The top-25-by-count list gets dominated by near-identical grays (background,
# body, anti-aliased clutter edges) — the actually diagnostic part colors
# (beak, foot/leg, tail) are smaller regions and get buried below that cutoff.
# This second pass filters specifically for *saturated* colors (channels far
# apart from each other, i.e. not gray/white/black) regardless of how small
# the pixel count is, which is exactly what a flat-colored 3D-rendered part
# looks like.
print("\n" + "=" * 42)
print("SATURATED COLORS ONLY (likely diagnostic parts)")
print("=" * 42)
saturated = [(c, n) for c, n in counts.items() if (max(c) - min(c)) > 40]
saturated.sort(key=lambda x: -x[1])
for color, count in saturated[:20]:
    pct = 100 * count / len(pixels)
    print(f"{str(color):<18} {count:<12} {pct:.2f}%")
