"""
IRIS-XAI — XAI Module

Provides a unified interface for three explanation methods:
  1. Grad-CAM          -> fast, CNN-specific, produces a spatial heatmap
  2. SHAP (Gradient)   -> pixel-level attributions, slower, model-agnostic-ish
  3. LIME              -> perturbation-based, treats the model as a black box

Each explainer takes a trained model + a single image and returns a
normalized importance map of shape (H, W) in [0, 1], so the Benchmark
Engine can compare methods on equal footing regardless of how each
library represents its output internally.

Usage:
    from xai_module import explain_gradcam, explain_shap, explain_lime

    heatmap = explain_gradcam(model, image_tensor, target_class=3, device="cpu")
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_map(arr: np.ndarray) -> np.ndarray:
    """Rescales any importance map to [0, 1] so all three methods are
    directly comparable regardless of their native output range."""
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def _model_device(model: torch.nn.Module) -> torch.device:
    """
    Returns whatever device the model's parameters are currently on.

    All three explainers used to force `model.to(device)` with a default
    of device="cpu" — since the Benchmark Engine's internal call site never
    passed a device through, this silently dragged an already-GPU model
    back onto the CPU on every single explanation call (in place, since
    nn.Module.to() mutates the model), corrupting it for every subsequent
    call in the same benchmark run that assumed the model was still on GPU.
    Reading the model's actual device instead of forcing a move sidesteps
    that entirely: whoever set up the model (e.g. BenchmarkEngine.run(),
    which already does model.to(device)) stays in control of where it lives.
    """
    return next(model.parameters()).device


def _find_last_conv_layer(model: torch.nn.Module):
    """
    Grad-CAM needs to hook into the last convolutional layer. Rather than
    hardcoding a layer name per architecture (which breaks the moment a new
    model is added to the Model Module), this walks the model and returns
    the last Conv2d layer found — works for SimpleCNN, ResNet, and
    EfficientNet without any per-model special-casing.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model — Grad-CAM requires a CNN.")
    return last_conv


# ---------------------------------------------------------------------------
# 1. Grad-CAM
# ---------------------------------------------------------------------------

def explain_gradcam(model, image_tensor, target_class: int, device: str = "cpu") -> np.ndarray:
    """
    Args:
        model: trained model (eval mode is set internally)
        image_tensor: single image, shape (1, 3, H, W)
        target_class: class index to explain

    Returns:
        (H, W) numpy array, normalized to [0, 1], resized to match input resolution.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    device = _model_device(model)  # match the model's actual device, don't move it
    image_tensor = image_tensor.to(device)

    target_layers = [_find_last_conv_layer(model)]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(target_class)])

    # grayscale_cam comes back as (batch, H, W); we only pass one image.
    return _normalize_map(grayscale_cam[0])


# ---------------------------------------------------------------------------
# 2. SHAP (Gradient-based)
# ---------------------------------------------------------------------------

def explain_shap(model, image_tensor, target_class: int, device: str = "cpu",
                  n_background: int = 5, image_size: int = None) -> np.ndarray:
    """
    Uses shap.GradientExplainer. Note: this is meaningfully slower than
    Grad-CAM (typically seconds rather than milliseconds per image) because
    it samples gradients across multiple background references — that's a
    genuine trade-off of the method, not a bug. If you're explaining many
    images in the Benchmark Engine, budget more time for this method
    specifically, or reduce image_size to speed it up.

    Args:
        n_background: number of background samples SHAP uses as its
                      reference distribution. Lower = faster but noisier
                      attributions; 5 is a reasonable default for quick runs.
        image_size: if set, downsamples the image before explaining (SHAP's
                    gradient sampling cost scales with image size). Leave
                    None to explain at full resolution.

    Returns:
        (H, W) numpy array, normalized to [0, 1].
    """
    import shap

    model.eval()
    device = _model_device(model)  # match the model's actual device, don't move it
    image_tensor = image_tensor.to(device)

    working_tensor = image_tensor
    if image_size is not None and image_tensor.shape[-1] != image_size:
        working_tensor = F.interpolate(image_tensor, size=(image_size, image_size), mode="bilinear")

    background = torch.zeros((n_background, *working_tensor.shape[1:]), device=device)
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(working_tensor)

    # SHAP's output format has changed across library versions:
    #   - older versions: a list of arrays, one per class, each (batch, C, H, W)
    #   - newer versions: a single array (batch, C, H, W, num_classes)
    # Both are handled here so this works regardless of which version pip
    # happened to install.
    if isinstance(shap_values, list):
        class_shap = shap_values[target_class]       # (batch, C, H, W)
        per_pixel = np.abs(class_shap[0]).sum(axis=0)  # -> (H, W)
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 5:
            class_shap = shap_values[..., target_class]  # (batch, C, H, W)
            per_pixel = np.abs(class_shap[0]).sum(axis=0)  # -> (H, W)
        else:
            # (batch, C, H, W) with no class dimension at all (single-output model)
            per_pixel = np.abs(shap_values[0]).sum(axis=0)  # -> (H, W)

    if image_size is not None:
        # Resize back up to the original resolution so all three methods
        # return maps at the same shape for fair comparison.
        per_pixel_tensor = torch.tensor(per_pixel).unsqueeze(0).unsqueeze(0)
        target_h, target_w = image_tensor.shape[-2:]
        per_pixel = F.interpolate(per_pixel_tensor, size=(target_h, target_w), mode="bilinear")[0, 0].numpy()

    return _normalize_map(per_pixel)


# ---------------------------------------------------------------------------
# 3. LIME
# ---------------------------------------------------------------------------

def explain_lime(model, image_tensor, target_class: int, device: str = "cpu",
                  num_samples: int = 60) -> np.ndarray:
    """
    Uses lime_image. LIME works on numpy images (H, W, 3) in [0, 1] rather
    than normalized tensors, so this function handles the conversion in
    both directions internally — the caller still just passes in the same
    normalized tensor used for the other two methods.

    Args:
        num_samples: number of perturbed samples LIME generates per
                     explanation. Lower = faster but noisier; 60 is tuned
                     for benchmark-scale runs where the same explanation
                     also gets regenerated 3x more for the stability metric
                     — at 200 samples, one LIME-explained image meant ~800
                     total perturbation runs once stability was included,
                     which is what made LIME the dominant cost in a full
                     benchmark pass. LIME's paper commonly uses 1000+
                     for production explanations, so treat this as a
                     fast/approximate setting suitable for benchmarking
                     many images, not a final publication-quality explanation.

    Returns:
        (H, W) numpy array, normalized to [0, 1].
    """
    from lime import lime_image

    model.eval()
    device = _model_device(model)  # match the model's actual device, don't move it

    # LIME's perturbation/segmentation logic expects a plain [0, 1] RGB
    # image, but `image_tensor` arrives here already normalized (e.g.
    # transforms.Normalize(CIFAR_MEAN, CIFAR_STD) or (0.5,0.5,0.5) for
    # FunnyBirds — both datasets normalize in dataset_module.py). Rescaling
    # to [0, 1] for LIME's own use is fine and necessary. The bug was that
    # predict_fn used to feed those rescaled [0, 1] perturbed images
    # straight back into the model with no re-normalization: the model was
    # trained on normalized inputs (roughly [-2, 2] for CIFAR, [-1, 1] for
    # FunnyBirds), so every prediction LIME's surrogate model was fit on
    # came from feeding the CNN out-of-distribution input — silently
    # degrading LIME's explanations relative to Grad-CAM/SHAP, which never
    # touch this rescale and stay in the model's expected input space.
    #
    # Fix: record the exact affine map used to go from the original
    # normalized tensor to [0, 1] (orig_min, orig_max), then invert that
    # same map inside predict_fn before calling the model — this works
    # regardless of which dataset's normalization stats were used upstream,
    # since it's undoing this function's own rescale rather than assuming
    # a specific mean/std.
    orig_min = image_tensor.min().item()
    orig_max = image_tensor.max().item()
    orig_range = (orig_max - orig_min) + 1e-8

    img = image_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = (img - orig_min) / orig_range

    def predict_fn(images_np):
        """LIME calls this repeatedly with batches of perturbed images."""
        batch = torch.tensor(images_np.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
        # Undo the [0, 1] rescale above to put perturbed samples back into
        # the same normalized space the model was trained on.
        batch = batch * orig_range + orig_min
        with torch.no_grad():
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    # top_labels defaults to 5 in LIME's own API, and when set (not None) it
    # silently OVERRIDES the labels= argument above — LIME just explains its
    # own top-5 predicted classes instead of the one you asked for. That was
    # the exact cause of every 'Label not in explanation' skip seen so far:
    # nothing to do with model accuracy, just this default quietly winning.
    # Setting top_labels=None makes LIME respect target_class every time.
    explanation = explainer.explain_instance(
        img, predict_fn, labels=(target_class,), top_labels=None,
        num_samples=num_samples, hide_color=0,
    )

    # get_image_and_mask returns a boolean/float mask over LIME's superpixel
    # segments; positive_only=False keeps both supporting and opposing
    # regions so the map reflects the full explanation, not just one side.
    _, mask = explanation.get_image_and_mask(
        target_class, positive_only=False, num_features=10, hide_rest=False
    )

    return _normalize_map(mask.astype(np.float32))


# ---------------------------------------------------------------------------
# Quick manual test — run this file directly to sanity-check all three methods
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from model_module import get_model

    device = "cpu"
    print(f"Using device: {device}")

    print("\nBuilding a pretrained ResNet18 for the smoke test...")
    model = get_model("resnet18", num_classes=10, pretrained=True)
    model.eval()

    # A random dummy image stands in for a real one here — this test only
    # confirms each explainer runs end-to-end without crashing, not that
    # the explanations are meaningful (that requires a real trained model
    # and real image, which the Benchmark Engine will use later).
    dummy_image = torch.rand(1, 3, 224, 224)
    target_class = 0

    print("\n" + "=" * 50)
    print("Testing Grad-CAM")
    print("=" * 50)
    heatmap = explain_gradcam(model, dummy_image, target_class, device=device)
    print(f"Output shape: {heatmap.shape} | min={heatmap.min():.3f} | max={heatmap.max():.3f}")

    print("\n" + "=" * 50)
    print("Testing SHAP (downsampled to 64x64 for smoke-test speed)")
    print("=" * 50)
    shap_map = explain_shap(model, dummy_image, target_class, device=device, n_background=3, image_size=64)
    print(f"Output shape: {shap_map.shape} | min={shap_map.min():.3f} | max={shap_map.max():.3f}")

    print("\n" + "=" * 50)
    print("Testing LIME (reduced samples for smoke-test speed)")
    print("=" * 50)
    lime_map = explain_lime(model, dummy_image, target_class, device=device, num_samples=50)
    print(f"Output shape: {lime_map.shape} | min={lime_map.min():.3f} | max={lime_map.max():.3f}")

    print("\nXAI Module sanity check complete.")
