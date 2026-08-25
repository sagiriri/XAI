"""
benchmark_module.py
====================
Benchmark Engine — ties Model Module + XAI Module together.

Computes two families of metrics for every (dataset x model x XAI method)
combination:

  1. MODEL performance metrics   -> Accuracy, Precision, Recall, F1, AUC,
                                     Inference Time, Model Size
  2. EXPLANATION quality metrics -> Faithfulness (deletion/insertion AUC),
                                     Stability/Robustness, Sparsity/Complexity,
                                     Explanation Runtime

Results are appended as rows to a pandas DataFrame and saved to CSV in
../results/, so the Visual Analytics module has a single structured file
to query instead of nothing (this was gap #2 from the original review).

Expected interfaces from your existing modules (adjust the imports below
if your actual function names differ slightly):

  model_module.py:
      evaluate_model(model, dataloader, device) -> dict with at least
          {'accuracy': float, 'avg_batch_inference_time_sec': float}
      get_model_size_mb(model) -> float

  xai_module.py:
      explain_gradcam(model, image_tensor, target_class) -> np.ndarray (H, W) in [0,1]
      explain_shap(model, image_tensor, target_class)    -> np.ndarray (H, W) in [0,1]
      explain_lime(model, image_tensor, target_class)    -> np.ndarray (H, W) in [0,1]

If any of these names/signatures don't match your files exactly, tell me
the real ones and I'll patch this file rather than you editing blind.
"""

import os
import time
import json
import functools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from model_module import evaluate_model, get_model_size_mb
from xai_module import explain_gradcam, explain_shap, explain_lime
from fidelity_module import paper_style_deletion_auc


def _trapz(y, x):
    """numpy renamed trapz -> trapezoid in 2.0; support both so this runs
    regardless of which numpy version is installed."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# ---------------------------------------------------------------------
# 1. MODEL PERFORMANCE METRICS
# ---------------------------------------------------------------------
def compute_model_metrics(model, dataloader, device, num_classes, max_batches=None):
    """
    Runs the model over dataloader once, computing Accuracy / Precision /
    Recall / F1 / AUC (macro, one-vs-rest) plus inference time and size.
    Reuses evaluate_model() for accuracy + inference time, and adds the
    classification metrics evaluate_model doesn't compute.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        # AUC undefined if a class is missing from this sample of batches
        auc = float("nan")

    base_metrics = evaluate_model(model, dataloader, device)
    size_mb = get_model_size_mb(model)

    return {
        "accuracy": base_metrics.get("accuracy", float((all_preds == all_labels).mean())),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "auc_macro_ovr": auc,
        "avg_batch_inference_time_sec": base_metrics.get("avg_batch_inference_time_sec"),
        "model_size_mb": size_mb,
    }


# ---------------------------------------------------------------------
# 2. EXPLANATION QUALITY METRICS
# ---------------------------------------------------------------------
def _explain(explain_fn, model, image_tensor, target_class):
    """
    Your XAI methods expect an already-batched (1, C, H, W) tensor (that's
    how xai_module.py's own smoke test called them). Everywhere else in
    this file, image_tensor is kept unbatched (C, H, W) for masking, so
    this wrapper adds the batch dim only at the explain_fn call site.
    """
    batched = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
    return explain_fn(model, batched, target_class)


def _predict_class_prob(model, image_tensor, target_class, device):
    """Returns model's softmax probability for target_class on a single image."""
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        prob = F.softmax(logits, dim=1)[0, target_class].item()
    return prob


def _patchify_importance(explanation, patch_size=16):
    """
    Aggregates a (H, W) explanation map into a coarser grid of patches and
    returns (patch_scores, n_patches_h, n_patches_w) so deletion/insertion
    can mask whole patches instead of single pixels (faster, less noisy).
    """
    H, W = explanation.shape
    ph, pw = H // patch_size, W // patch_size
    trimmed = explanation[: ph * patch_size, : pw * patch_size]
    patches = trimmed.reshape(ph, patch_size, pw, patch_size)
    patch_scores = patches.mean(axis=(1, 3))  # (ph, pw)
    return patch_scores, ph, pw


def _apply_patch_mask(image_tensor, patch_order, n_patches_to_mask, ph, pw,
                       patch_size, baseline_value, keep_mode="delete"):
    """
    keep_mode='delete'  -> mask the top-n_patches_to_mask most important patches
    keep_mode='insert'  -> start blank, reveal the top-n_patches_to_mask patches
    """
    masked = image_tensor.clone()
    if keep_mode == "delete":
        target_patches = patch_order[:n_patches_to_mask]
        for idx in target_patches:
            r, c = idx // pw, idx % pw
            masked[:, r * patch_size:(r + 1) * patch_size,
                      c * patch_size:(c + 1) * patch_size] = baseline_value
    else:  # insert
        canvas = torch.full_like(image_tensor, baseline_value)
        target_patches = patch_order[:n_patches_to_mask]
        for idx in target_patches:
            r, c = idx // pw, idx % pw
            canvas[:, r * patch_size:(r + 1) * patch_size,
                      c * patch_size:(c + 1) * patch_size] = \
                image_tensor[:, r * patch_size:(r + 1) * patch_size,
                                c * patch_size:(c + 1) * patch_size]
        masked = canvas
    return masked


def faithfulness_deletion_insertion_auc(model, image_tensor, target_class, explanation,
                                         device, patch_size=16, steps=10):
    """
    Deletion: progressively mask the most important patches, track predicted
    probability of target_class. Fast drop = faithful -> LOW deletion AUC is good.
    Insertion: progressively reveal the most important patches on a blank
    canvas. Fast rise = faithful -> HIGH insertion AUC is good.
    """
    patch_scores, ph, pw = _patchify_importance(explanation, patch_size)
    flat_scores = patch_scores.flatten()
    patch_order = np.argsort(-flat_scores)  # most important first
    total_patches = len(patch_order)
    baseline_value = image_tensor.mean().item()

    fractions = np.linspace(0, 1, steps)
    del_probs, ins_probs = [], []

    for frac in fractions:
        n_patches = int(round(frac * total_patches))

        del_img = _apply_patch_mask(image_tensor, patch_order, n_patches, ph, pw,
                                     patch_size, baseline_value, keep_mode="delete")
        del_probs.append(_predict_class_prob(model, del_img, target_class, device))

        ins_img = _apply_patch_mask(image_tensor, patch_order, n_patches, ph, pw,
                                     patch_size, baseline_value, keep_mode="insert")
        ins_probs.append(_predict_class_prob(model, ins_img, target_class, device))

    deletion_auc = float(_trapz(del_probs, fractions))
    insertion_auc = float(_trapz(ins_probs, fractions))
    return deletion_auc, insertion_auc


def stability_score(model, image_tensor, target_class, explain_fn, device,
                     n_perturbations=3, noise_std=0.05, original_explanation=None):
    """
    Adds small Gaussian noise to the input n_perturbations times, regenerates
    the explanation each time, and compares to the original explanation via
    cosine similarity (higher/closer to 1 = more stable) and max-sensitivity
    (largest observed change, lower = more robust).

    original_explanation: pass in the explanation already computed by the
        caller (compute_explanation_metrics always has one on hand) instead
        of recomputing it here. For LIME especially, each explanation means
        regenerating 200 perturbed superpixel samples from scratch — a
        redundant recompute here was silently adding ~15% pure waste on top
        of an already-expensive method. n_perturbations was also dropped
        from 5 to 3: still enough for a stable mean/max, at 40% less cost.
    """
    if original_explanation is None:
        original_explanation = _explain(explain_fn, model, image_tensor, target_class)
    orig_flat = original_explanation.flatten()
    orig_norm = np.linalg.norm(orig_flat) + 1e-8

    sims, sensitivities = [], []
    for _ in range(n_perturbations):
        noise = torch.randn_like(image_tensor) * noise_std
        perturbed = torch.clamp(image_tensor + noise, image_tensor.min(), image_tensor.max())
        perturbed_explanation = _explain(explain_fn, model, perturbed, target_class)
        pert_flat = perturbed_explanation.flatten()

        cos_sim = float(np.dot(orig_flat, pert_flat) /
                         (orig_norm * (np.linalg.norm(pert_flat) + 1e-8)))
        sims.append(cos_sim)

        sensitivity = float(np.linalg.norm(pert_flat - orig_flat) / orig_norm)
        sensitivities.append(sensitivity)

    return {
        "stability_cosine_sim": float(np.mean(sims)),
        "max_sensitivity": float(np.max(sensitivities)),
    }


def sparsity_complexity(explanation, eps=1e-8):
    """
    Shannon-entropy-based complexity: treats the normalized explanation map
    as a probability distribution and computes its entropy. LOWER entropy
    means the explanation concentrates on fewer regions (more sparse, more
    usable by a human) -- this is the standard "Complexity" metric used in
    XAI-benchmarking literature (e.g. Quantus).
    """
    flat = np.abs(explanation.flatten())
    total = flat.sum() + eps
    p = flat / total
    entropy = float(-np.sum(p * np.log(p + eps)))
    max_entropy = float(np.log(len(flat)))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else float("nan")
    return {
        "complexity_entropy": entropy,
        "complexity_entropy_normalized": normalized_entropy,
    }


def compute_explanation_metrics(model, image_tensor, target_class, explain_fn, device,
                                  part_map_path=None):
    """Runs one XAI method on one image and returns every explanation-quality metric.

    If part_map_path is given (FunnyBirds only), also scores the explanation
    against real ground truth: does it actually point at a bird part
    (beak/eye/foot/wing) or at background clutter? Reuses the already-computed
    explanation rather than re-running the (possibly slow) XAI method again.
    """
    start = time.time()
    explanation = _explain(explain_fn, model, image_tensor, target_class)
    runtime_sec = time.time() - start

    deletion_auc, insertion_auc = faithfulness_deletion_insertion_auc(
        model, image_tensor, target_class, explanation, device
    )
    paper_deletion_auc = paper_style_deletion_auc(
    model=model,
    image_tensor=image_tensor,
    target_class=target_class,
    explanation=explanation,
    device=device,
)
    stability = stability_score(model, image_tensor, target_class, explain_fn, device,
                                  original_explanation=explanation)
    complexity = sparsity_complexity(explanation)

    result = {
        "explanation_runtime_sec": runtime_sec,
        "faithfulness_deletion_auc": deletion_auc,   # lower is better
        "faithfulness_insertion_auc": insertion_auc, # higher is better
         "paper_deletion_auc": paper_deletion_auc,
        **stability,
        **complexity,
    }

    if part_map_path is not None:
        from funnybirds_ground_truth import ground_truth_part_score
        try:
            gt = ground_truth_part_score(explanation, part_map_path)
            result["part_overlap_ratio"] = gt["part_overlap_ratio"]
            result["clutter_leakage_ratio"] = gt["clutter_leakage_ratio"]
        except Exception as e:
            print(f"     [ground-truth part score failed for this sample: {e}]")

    return result


# ---------------------------------------------------------------------
# 3. BENCHMARK RUNNER
# ---------------------------------------------------------------------
XAI_METHODS = {
    "gradcam": explain_gradcam,
    # SHAP at full 224x224 resolution is what caused the earlier hang —
    # gradient sampling across every pixel of a full-res image, repeated
    # per background sample, is exactly the expensive case that triggered
    # it. Downsampling to 64x64 first (the same setting that worked
    # cleanly in the original xai_module.py smoke test) keeps SHAP's
    # output the same shape as the other two methods (still resized back
    # up to full resolution internally) while cutting the actual gradient
    # computation to a small fraction of the work.
    "shap": functools.partial(explain_shap, image_size=64),
    "lime": explain_lime,
}


class BenchmarkEngine:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.rows = []

    def run(self, models: dict, dataloader, dataset_name: str, device,
             num_classes: int, num_xai_samples: int = 10,
             xai_methods: dict = None, model_max_batches: int = None,
             xai_samples: list = None, output_filename: str = None):
        """
        models        : {"resnet18": model_instance, "efficientnet": model_instance, ...}
        dataloader    : test-set DataLoader for `dataset_name` — always plain
                        (images, labels), used for model metrics (accuracy etc).
        num_xai_samples: how many test images to run each XAI method on
                          (kept small by default since SHAP/LIME are slow on CPU)
        xai_methods   : subset of XAI_METHODS to run; defaults to all three
        xai_samples   : optional explicit list of (image_tensor, label, part_map_path)
                         tuples to explain instead of sampling from `dataloader`.
                         Used for FunnyBirds so each sample carries its
                         ground-truth part-map path; leave None for CIFAR-10.
        output_filename : if given, saves the CSV after EVERY model finishes,
                           not just once at the very end. A run covering 3
                           models can take hours (LIME's stability check
                           especially); without this, a Ctrl+C or crash
                           partway through silently discards every model
                           that had already finished. Pass the same filename
                           you'd otherwise give to .save() afterward.
        """
        xai_methods = xai_methods or XAI_METHODS

        # Pull a fixed pool of sample images/labels once, reused across models.
        # If xai_samples is given explicitly (FunnyBirds, with part_map paths
        # attached), use that directly. Otherwise fall back to sampling from
        # `dataloader` as before — this keeps `dataloader` itself always a
        # plain (images, labels) loader, since compute_model_metrics() and
        # evaluate_model() both hard-unpack 2-tuples and would break on the
        # 3-tuple (images, labels, paths) format used elsewhere for ground truth.
        if xai_samples is not None:
            sample_images = [s[0] for s in xai_samples]
            sample_labels = [int(s[1]) for s in xai_samples]
            sample_paths = [s[2] for s in xai_samples]
        else:
            sample_images, sample_labels, sample_paths = [], [], []
            for images, labels in dataloader:
                for img, lbl in zip(images, labels):
                    sample_images.append(img)
                    sample_labels.append(int(lbl))
                    sample_paths.append(None)
                    if len(sample_images) >= num_xai_samples:
                        break
                if len(sample_images) >= num_xai_samples:
                    break

        for model_name, model in models.items():
            print(f"\n{'='*60}\nBenchmarking model: {model_name} on {dataset_name}\n{'='*60}")
            model.to(device)

            model_metrics = compute_model_metrics(
                model, dataloader, device, num_classes, max_batches=model_max_batches
            )
            print(f"  Model metrics: {model_metrics}")

            for method_name, explain_fn in xai_methods.items():
                print(f"  -- XAI method: {method_name} "
                      f"({len(sample_images)} samples) --")
                per_sample_rows = []
                for img, label, path in zip(sample_images, sample_labels, sample_paths):
                    try:
                        m = compute_explanation_metrics(model, img, label, explain_fn, device,
                                                          part_map_path=path)
                        per_sample_rows.append(m)
                    except Exception as e:
                        print(f"     [skipped one sample due to error: {e}]")

                if not per_sample_rows:
                    print(f"     WARNING: no samples succeeded for {method_name}, skipping.")
                    continue

                agg = pd.DataFrame(per_sample_rows).mean(numeric_only=True).to_dict()

                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "xai_method": method_name,
                    "n_xai_samples": len(per_sample_rows),
                    **model_metrics,
                    **agg,
                }
                self.rows.append(row)
                print(f"     -> {method_name} aggregated: "
                      f"deletion_auc={agg.get('faithfulness_deletion_auc'):.4f}, "
                      f"insertion_auc={agg.get('faithfulness_insertion_auc'):.4f}, "
                      f"stability={agg.get('stability_cosine_sim'):.4f}, "
                      f"runtime={agg.get('explanation_runtime_sec'):.3f}s")

            if output_filename is not None:
                self.save(output_filename)
                print(f"  [checkpoint] saved progress through '{model_name}' to {output_filename}")

        return self.rows

    def save(self, filename="benchmark_results.csv"):
        path = os.path.join(self.results_dir, filename)
        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)
        print(f"\nSaved {len(df)} result rows to {path}")
        return path


# ---------------------------------------------------------------------
# SMOKE TEST (kept for reference — run with --smoketest)
# ---------------------------------------------------------------------
def run_smoketest():
    from dataset_module import get_cifar_loaders
    from model_module import get_model

    device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, classes = get_cifar_loaders(batch_size=8)
    num_classes = len(classes)

    models = {
        "simplecnn": get_model("simplecnn", num_classes=num_classes, pretrained=False),
        "resnet18": get_model("resnet18", num_classes=num_classes, pretrained=True),
    }

    engine = BenchmarkEngine(results_dir="results")
    engine.run(
        models=models,
        dataloader=test_loader,
        dataset_name="CIFAR10",
        device=device,
        num_classes=num_classes,
        num_xai_samples=3,          # keep tiny for the smoke test
        model_max_batches=5,        # keep tiny for the smoke test
    )
    engine.save("benchmark_results_smoketest.csv")
    print("\nBenchmark Engine sanity check complete.")


# ---------------------------------------------------------------------
# REAL RUN — loads your trained checkpoints from train_full.py instead of
# building fresh (untrained) models. This is what produces report-ready
# numbers instead of ~10%-accuracy smoke-test placeholders.
# ---------------------------------------------------------------------
def run_from_checkpoints(dataset_name, model_names, num_xai_samples, checkpoint_dir, output_filename,
                          xai_method_names=None):
    from dataset_module import get_cifar_loaders, get_funnybirds_loaders
    from model_module import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if dataset_name == "cifar10":
        _, test_loader, classes = get_cifar_loaders(
            dataset_name="cifar10", data_dir="../data", batch_size=32, image_size=224, num_workers=0
        )
        xai_samples = None  # falls back to sampling from test_loader inside engine.run()
    elif dataset_name == "funnybirds":
        _, test_loader, classes = get_funnybirds_loaders(
            data_dir="../data/FunnyBirds", batch_size=32, image_size=224, num_workers=0
        )
        # A second, path-aware view of the same test set — used only to pull
        # the small XAI sample pool with each image's ground-truth part_map
        # path attached. test_loader itself stays plain (images, labels) so
        # compute_model_metrics()/evaluate_model() aren't affected.
        _, test_loader_with_paths, _ = get_funnybirds_loaders(
            data_dir="../data/FunnyBirds", batch_size=1, image_size=224, num_workers=0,
            return_paths=True,
        )
        xai_samples = []
        for image, label, path in test_loader_with_paths:
            xai_samples.append((image[0], int(label[0]), path[0]))
            if len(xai_samples) >= num_xai_samples:
                break
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    num_classes = len(classes)

    models = {}
    for model_name in model_names:
        ckpt_path = os.path.join(checkpoint_dir, f"{dataset_name}_{model_name}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [skip] no checkpoint found at {ckpt_path} — train it first with train_full.py")
            continue
        print(f"  Loading checkpoint: {ckpt_path}")
        models[model_name] = load_model(model_name, ckpt_path, num_classes=num_classes, device=device)

    if not models:
        print("No checkpoints found — nothing to benchmark. Run train_full.py first.")
        return

    engine = BenchmarkEngine(results_dir="results")
    xai_methods = ({k: XAI_METHODS[k] for k in xai_method_names}
                    if xai_method_names else None)
    engine.run(
        models=models,
        dataloader=test_loader,
        dataset_name=dataset_name.upper(),
        device=device,
        num_classes=num_classes,
        num_xai_samples=num_xai_samples,
        model_max_batches=None,     # full test set now that we have a real GPU + real model
        xai_samples=xai_samples,    # ground-truth-aware samples for FunnyBirds, None for CIFAR-10
        output_filename=output_filename,  # saves after every model, not just at the end
        xai_methods=xai_methods,
    )
    engine.save(output_filename)
    print(f"\nBenchmark run on trained checkpoints complete: {output_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoketest", action="store_true",
                         help="run the original tiny CPU sanity check instead of real checkpoints")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "funnybirds"])
    parser.add_argument("--models", nargs="+", default=["resnet18", "efficientnet"],
                         choices=["simplecnn", "resnet18", "resnet50", "efficientnet"],
                         help="which trained checkpoints to benchmark")
    parser.add_argument("--num_xai_samples", type=int, default=15,
                         help="how many test images to run each XAI method on. "
                              "SHAP is the slow one — keep this modest for a same-day run, "
                              "raise it later for final report numbers.")
    parser.add_argument("--xai_methods", nargs="+", default=["gradcam", "shap", "lime"],
                         choices=["gradcam", "shap", "lime"],
                         help="which XAI methods to run. SHAP's GradientExplainer has been "
                              "unreliable on some torch/CUDA combinations (can hang "
                              "indefinitely with no error). If it hangs, rerun with "
                              "--xai_methods gradcam lime to skip it.")
    parser.add_argument("--checkpoint_dir", default="../models")
    parser.add_argument("--output", default=None,
                         help="output CSV filename; defaults to benchmark_results_<dataset>.csv")
    args = parser.parse_args()

    if args.smoketest:
        run_smoketest()
    else:
        output_filename = args.output or f"benchmark_results_{args.dataset}.csv"
        run_from_checkpoints(
            dataset_name=args.dataset,
            model_names=args.models,
            num_xai_samples=args.num_xai_samples,
            checkpoint_dir=args.checkpoint_dir,
            xai_method_names=args.xai_methods,
            output_filename=output_filename,
        )
