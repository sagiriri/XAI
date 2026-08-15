"""
XAIBench — FunnyBirds Ground-Truth Part Metric

Every faithfulness/stability/complexity metric in the Benchmark Engine so
far is *self-referential* — it checks whether an explanation is internally
consistent (e.g. does confidence drop when the explanation's own top pixels
are removed), never whether it's actually pointing at the right thing.

FunnyBirds' train_part_map/ and test_part_map/ folders give us real ground
truth to check against: each part (beak, eye, foot, wing) is rendered in a
fixed canonical color, verified empirically by scanning 45 images across 15
classes (see inspect_part_colors.py / scan_all_part_colors.py). Tail was
excluded — it never appeared as a distinct color in any sampled image,
likely occluded by camera framing, so it's not part of this ground truth.

This module measures: of the pixels an explanation method marks as
important, what fraction actually land on a real bird part vs. on
background clutter (the scattered gray distractor shapes FunnyBirds
deliberately includes to test whether a model — and, here, an explanation
— is distracted by irrelevant objects)?

Usage:
    from funnybirds_ground_truth import ground_truth_part_score

    score = ground_truth_part_score(
        explanation_map=heatmap,          # (H, W) array in [0, 1], from any XAI method
        part_map_path="../data/FunnyBirds/train_part_map/3/003000.png",
    )
    # -> {"part_overlap_ratio": 0.62, "clutter_leakage_ratio": 0.08, ...}
"""

import numpy as np
from PIL import Image


# Canonical colors, verified empirically (see inspect_part_colors.py output).
# Small anti-aliased/faded variants of each are matched too, not just the
# exact pure value, since rendered edges blend toward neighboring colors.
CANONICAL_PART_COLORS = {
    "beak": (255, 255, 0),
    "eye": (0, 0, 255),
    "foot": (255, 0, 1),
    "wing": (0, 255, 1),
}
BODY_COLOR = (170, 170, 170)          # flat gray — the bird's own body, not a distractor
BACKGROUND_COLOR = (0, 0, 0)          # pure black
CLUTTER_MIN_GRAY = 190                # the scattered distractor blobs render as light gray/white
CLUTTER_MAX_GRAY = 255
SATURATION_THRESHOLD = 40             # max(rgb) - min(rgb); anything above this is a "colored" pixel


def _classify_pixel(r, g, b):
    """
    Returns one of: "beak", "eye", "foot", "wing", "body", "clutter", "background".

    Faded/anti-aliased pixels near a part's edge get matched to that part by
    checking which canonical color's nonzero-channel pattern they follow
    (e.g. a dim red like (128,0,1) still has "only the R channel nonzero",
    same pattern as pure red foot), rather than requiring an exact match.
    """
    # Colors arrive as numpy uint8 — subtracting them directly wraps around
    # instead of going negative (e.g. 204-220 becomes 240, not -16), which
    # silently corrupts every comparison below. Cast to plain int first.
    r, g, b = int(r), int(g), int(b)

    if (r, g, b) == BACKGROUND_COLOR:
        return "background"

    saturated = (max(r, g, b) - min(r, g, b)) > SATURATION_THRESHOLD
    if saturated:
        if r > 0 and g == 0 and b <= 2:
            return "foot"
        if r == 0 and g > 0 and b <= 2:
            return "wing"
        if r == 0 and g == 0 and b > 0:
            return "eye"
        if r > 0 and g > 0 and abs(int(r) - int(g)) < 10 and b <= 2:
            return "beak"
        # A saturated color that doesn't match any known part pattern —
        # most likely an edge-blend artifact between two regions. Treat as
        # body/clutter-adjacent noise rather than a phantom 5th part.
        return "body"

    # Unsaturated (grayish) pixel — distinguish the bird's own body from the
    # scattered clutter distractor blobs by gray level.
    if CLUTTER_MIN_GRAY <= r <= CLUTTER_MAX_GRAY and abs(r - g) < 15 and abs(g - b) < 40:
        return "clutter"
    return "body"


def load_part_masks(part_map_path: str):
    """
    Loads a train_part_map / test_part_map image and returns a dict of
    boolean (H, W) numpy masks, one per category:
        "beak", "eye", "foot", "wing"  — the 4 verified diagnostic parts
        "body"                          — the bird's own silhouette (non-diagnostic)
        "clutter"                       — scattered distractor blobs (should NOT be attended to)
        "background"                    — empty space
    """
    img = Image.open(part_map_path).convert("RGB")
    arr = np.array(img)  # (H, W, 3)
    h, w, _ = arr.shape

    labels = np.empty((h, w), dtype=object)
    # Vectorized-ish: classify unique colors once, then map — much faster
    # than calling _classify_pixel per-pixel on a 256x256+ image.
    flat = arr.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    color_labels = np.array([_classify_pixel(*c) for c in unique_colors])
    label_flat = color_labels[inverse]
    labels = label_flat.reshape(h, w)

    masks = {}
    for category in ["beak", "eye", "foot", "wing", "body", "clutter", "background"]:
        masks[category] = (labels == category)
    return masks


def ground_truth_part_score(explanation_map: np.ndarray, part_map_path: str,
                             top_k_percent: float = 20.0) -> dict:
    """
    Args:
        explanation_map: (H, W) array in [0, 1], any XAI method's output
                          (already normalized — matches what explain_gradcam
                          / explain_shap / explain_lime return).
        part_map_path: path to the matching train_part_map/<class>/<file>.png
        top_k_percent: what fraction of pixels (by importance) count as
                        "the explanation's important region". 20% is a
                        reasonable default — roughly matches how much of
                        the frame a bird typically occupies.

    Returns:
        {
            "part_overlap_ratio": fraction of the explanation's important
                pixels that land on a real diagnostic part (beak/eye/foot/wing).
                Higher is better — the explanation is pointing at the bird's
                actual identifying features.
            "clutter_leakage_ratio": fraction landing on background clutter
                distractor objects. Lower is better — high values mean the
                explanation is (wrongly) attending to irrelevant objects.
            "part_breakdown": dict of per-part overlap fractions, so you can
                see e.g. "this method always finds the beak but never the wing".
        }
    """
    masks = load_part_masks(part_map_path)

    # Resize explanation_map to match the part map's resolution if they differ
    # (e.g. Grad-CAM sometimes returns a slightly different resolution than
    # the original image before normalization/resizing back up).
    target_h, target_w = masks["beak"].shape
    if explanation_map.shape != (target_h, target_w):
        exp_img = Image.fromarray((explanation_map * 255).astype(np.uint8))
        exp_img = exp_img.resize((target_w, target_h), Image.BILINEAR)
        explanation_map = np.array(exp_img).astype(np.float32) / 255.0

    threshold = np.percentile(explanation_map, 100 - top_k_percent)
    important = explanation_map >= threshold
    important_count = important.sum()

    # np.percentile ties break down when the map is mostly identical/zero
    # values (common — e.g. LIME masks, hard-thresholded Grad-CAM): the
    # percentile itself can land on 0, which then matches almost every
    # pixel instead of just the intended top slice, silently diluting every
    # ratio below. Selecting an exact pixel COUNT via argpartition sidesteps
    # ties entirely and always returns exactly top_k_percent of pixels.
    total_pixels = explanation_map.size
    k = max(1, int(round(total_pixels * top_k_percent / 100.0)))
    flat_exp = explanation_map.flatten()
    top_k_indices = np.argpartition(flat_exp, -k)[-k:]
    important_flat = np.zeros(total_pixels, dtype=bool)
    important_flat[top_k_indices] = True
    important = important_flat.reshape(explanation_map.shape)
    important_count = important.sum()

    if important_count == 0:
        return {"part_overlap_ratio": 0.0, "clutter_leakage_ratio": 0.0,
                "part_breakdown": {p: 0.0 for p in ["beak", "eye", "foot", "wing"]}}

    parts_mask = masks["beak"] | masks["eye"] | masks["foot"] | masks["wing"]
    part_overlap_ratio = (important & parts_mask).sum() / important_count
    clutter_leakage_ratio = (important & masks["clutter"]).sum() / important_count

    part_breakdown = {}
    for part in ["beak", "eye", "foot", "wing"]:
        part_breakdown[part] = float((important & masks[part]).sum() / important_count)

    return {
        "part_overlap_ratio": float(part_overlap_ratio),
        "clutter_leakage_ratio": float(clutter_leakage_ratio),
        "part_breakdown": part_breakdown,
    }


# ---------------------------------------------------------------------------
# Quick manual test — run this file directly to sanity-check on a real image
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob

    test_path = r"D:\XAI\data\FunnyBirds\train_part_map\3\003000.png"
    print(f"Testing on: {test_path}\n")

    masks = load_part_masks(test_path)
    print("Pixel counts per category:")
    for category, mask in masks.items():
        print(f"  {category:<12} {mask.sum()} pixels ({100 * mask.sum() / mask.size:.2f}%)")

    print("\nSanity-checking with a fake explanation (random noise) —")
    print("should show roughly proportional overlap, since a random map")
    print("has no reason to specifically target parts:")
    fake_explanation = np.random.rand(*masks["beak"].shape)
    result = ground_truth_part_score(fake_explanation, test_path)
    print(result)

    print("\nSanity-checking with a 'perfect' explanation (only the parts")
    print("themselves, at maximum importance) — should show ~100% overlap:")
    parts_mask = masks["beak"] | masks["eye"] | masks["foot"] | masks["wing"]
    perfect_explanation = parts_mask.astype(np.float32)
    result = ground_truth_part_score(perfect_explanation, test_path)
    print(result)

    print("\nfunnybirds_ground_truth.py sanity check complete.")
