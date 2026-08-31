"""
One-off inspection script — scans many train_part_map images at once to
settle whether a 5th canonical part color exists (for "tail"), since it
never appeared in the single image checked so far — likely occluded behind
the body/wing from that particular camera angle, not colorless.

Run with iris-xai-env active:
    python scan_all_part_colors.py
"""

import os
import glob
from collections import Counter
from PIL import Image

PART_MAP_DIR = r"D:\XAI\data\FunnyBirds\train_part_map"
IMAGES_PER_CLASS = 3   # keep this small — we're just hunting for the palette, not doing real work
MAX_CLASSES = 15       # scan across enough different classes/poses to catch an occluded part

# The 4 canonical colors already confirmed from the first image.
KNOWN_COLORS = {
    (0, 255, 1): "wing", (0, 255, 2): "wing",
    (0, 0, 255): "eye",
    (255, 255, 0): "beak",
    (255, 0, 1): "foot", (255, 0, 2): "foot",
}


def classify(color):
    """Buckets a saturated color into one of the known parts if it's a
    scaled (anti-aliased) version of one of them, else returns 'UNKNOWN'."""
    if color in KNOWN_COLORS:
        return KNOWN_COLORS[color]
    r, g, b = color
    # A scaled-down version of pure green/red/blue/yellow keeps the same
    # zero/nonzero channel pattern, just dimmer — e.g. (128,0,1) is a faded red.
    if r > 0 and g == 0 and b <= 2:
        return "foot (faded)"
    if r == 0 and g > 0 and b <= 2:
        return "wing (faded)"
    if r == 0 and g == 0 and b > 0:
        return "eye (faded)"
    if r > 0 and g > 0 and r == g and b <= 2:
        return "beak (faded)"
    return "UNKNOWN"


unknown_colors = Counter()
class_dirs = sorted(glob.glob(os.path.join(PART_MAP_DIR, "*")))[:MAX_CLASSES]

print(f"Scanning {len(class_dirs)} classes, {IMAGES_PER_CLASS} images each...\n")

for class_dir in class_dirs:
    images = sorted(glob.glob(os.path.join(class_dir, "*.png")))[:IMAGES_PER_CLASS]
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        pixels = list(img.getdata())
        counts = Counter(pixels)
        for color, count in counts.items():
            if (max(color) - min(color)) > 40:  # saturated only
                label = classify(color)
                if label == "UNKNOWN":
                    unknown_colors[color] += count

print("=" * 50)
print("UNKNOWN saturated colors found (not matching wing/eye/beak/foot)")
print("=" * 50)
if not unknown_colors:
    print("None found. Tail is very likely just occluded in every sampled")
    print("image, or it genuinely renders as the body's gray with no")
    print("distinct color. Worth checking test_part_map/ too, or just")
    print("proceeding with a 4-part ground truth (beak/eye/foot/wing) and")
    print("noting the tail limitation honestly in the report.")
else:
    for color, count in unknown_colors.most_common(15):
        print(f"{color}  (seen {count} times total across scanned images)")
