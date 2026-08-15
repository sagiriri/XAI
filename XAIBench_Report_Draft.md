# XAIBench: An Explainability Benchmark Platform for Image Classification

*Draft — results tables marked `[PENDING: 60-sample run]` will be filled in once the final benchmark completes.*

---

## 1. Introduction & Motivation

Explainable AI (XAI) methods like Grad-CAM, SHAP, and LIME are widely used to interpret deep learning models, but they are rarely evaluated against each other systematically. Most projects that use XAI pick one method and demonstrate it qualitatively — this project instead builds a **benchmark platform** that trains multiple models, applies multiple XAI methods to each, and scores every combination on a consistent set of quantitative metrics.

This follows the recent trend in the XAI literature of building standardized benchmark suites rather than one-off demonstrations — e.g. Saliency-Bench (2023), ExplainTS (2026), BEExAI (2025), and XAI-Units (2025) all take this approach. XAIBench applies the same methodology to a smaller, self-contained image-classification setting.

## 2. Datasets

| Dataset | Purpose | Classes | Notes |
|---|---|---|---|
| CIFAR-10 | Model training/evaluation, standard baseline | 10 | 50,000 train / 10,000 test |
| FunnyBirds | Ground-truth explanation evaluation | 50 (synthetic birds) | 50,000 train / 500 test; each image ships with a matching part-map that identifies exactly where the beak, eye, foot, and wing are — real ground truth for checking whether an explanation is pointing at the correct region, not just a plausible-looking one |

FunnyBirds is a synthetic, part-based dataset purpose-built for evaluating visual explanations (Hesse et al., 2023) — this is what differentiates XAIBench from a plain model-accuracy comparison: it lets us measure whether an explanation is *correct*, not just whether it looks reasonable.

## 3. Architecture

Five modules, each independently testable:

```
Dataset Module → Model Module → XAI Module → Benchmark Engine → Visual Analytics
                                      ↑
                        FunnyBirds Ground-Truth Module
                    (verifies explanations against real part locations)
```

- **Dataset Module** — loads CIFAR-10 and FunnyBirds with matching preprocessing; FunnyBirds loader can optionally return each sample's ground-truth part-map path alongside the image.
- **Model Module** — a unified `get_model()`/`train_model()`/`evaluate_model()` interface across three architectures: a custom SimpleCNN baseline, ResNet18, and EfficientNet-B0 (both ImageNet-pretrained).
- **XAI Module** — Grad-CAM, SHAP (GradientExplainer), and LIME, all normalized to the same `(H, W)` output format so they're directly comparable.
- **Benchmark Engine** — computes model performance (accuracy/precision/recall/F1/AUC), explanation faithfulness (deletion/insertion AUC), stability (robustness to input noise), complexity (entropy-based conciseness), and runtime, for every (model × XAI method) pair.
- **FunnyBirds Ground-Truth Module** — the project's key differentiator. Verifies each explanation's most important pixels against the dataset's real part locations, producing a *part overlap ratio* (does the explanation hit a real bird part?) and *clutter leakage ratio* (does it waste attention on background distractor objects?).
- **Visual Analytics** — an interactive dashboard presenting all of the above, with a composite Explainability Score and per-dataset filtering.

## 4. Methodology

### 4.1 Training
All three models trained for 22 epochs on both datasets, on GPU (RTX 3050), using Adam optimization and mixed precision. ResNet18/EfficientNet used ImageNet-pretrained weights; SimpleCNN trained from scratch as a lower-complexity baseline.

### 4.2 Ground-truth part-color verification
Before building the ground-truth metric, the canonical part colors used in FunnyBirds' segmentation renders were verified empirically rather than assumed — scanning pixel color distributions across 45 images spanning 15 classes confirmed four consistent, saturation-distinguishable colors:

| Part | Canonical color |
|---|---|
| Beak | Pure yellow `(255,255,0)` |
| Eye | Pure blue `(0,0,255)` |
| Foot | Pure red `(255,0,~1)` |
| Wing | Pure green `(0,255,~1)` |

A fifth part, **tail**, was checked for but never found as a distinct color across the sampled images — most likely occluded by typical camera framing rather than uncolored. Tail is therefore excluded from the ground-truth metric; this is a stated limitation rather than an assumption (see §7).

### 4.3 Explainability Score (composite metric)
For each (model × XAI method), five normalized metrics are averaged into a single 0–100 score: insertion AUC, (1 − deletion AUC), stability, (1 − max-sensitivity), and (1 − entropy). Each is min–max normalized across the run before averaging.

## 5. Engineering Challenges (worth documenting — shows the debugging process, not just the result)

A few non-trivial bugs surfaced during development that are worth noting for methodological transparency:

- **Silent device corruption in the XAI Module.** `explain_gradcam`/`explain_shap`/`explain_lime` originally defaulted to `device="cpu"` and force-moved the model there as a side effect, since the Benchmark Engine never explicitly passed a device through. This silently dragged an already-GPU model back to CPU mid-run, corrupting every subsequent computation in that pass. Fixed by having each explainer read the model's *actual* current device instead of forcing a move.
- **LIME's `top_labels` default silently overriding explicit label requests.** `LimeImageExplainer.explain_instance()` defaults `top_labels=5`, which — when set — overrides an explicitly passed `labels=` argument entirely. This caused `'Label not in explanation'` errors on a meaningful fraction of samples across every model and dataset tested, incorrectly appearing as a model-quality issue rather than an API default. Fixed by passing `top_labels=None`.
- **SHAP hanging at full resolution.** `shap.GradientExplainer` at full 224×224 resolution combined with the project's GPU/torch version combination caused indefinite hangs (confirmed via zero CPU/GPU utilization for 20+ minutes with no error). Downsampling to 64×64 before running SHAP (matching an earlier working configuration) resolved this without materially changing explanation quality, since the output is resized back to full resolution afterward.

## 6. Results

*[PENDING: 60-sample run — tables below will be populated with final numbers]*

### 6.1 Model Performance

| Model | Dataset | Accuracy | F1 | AUC | Params (MB) |
|---|---|---|---|---|---|
| SimpleCNN | CIFAR-10 | — | — | — | 1.50 |
| ResNet18 | CIFAR-10 | — | — | — | 42.65 |
| EfficientNet | CIFAR-10 | — | — | — | 15.34 |
| SimpleCNN | FunnyBirds | — | — | — | 1.53 |
| ResNet18 | FunnyBirds | — | — | — | 42.73 |
| EfficientNet | FunnyBirds | — | — | — | 15.53 |

### 6.2 Explainability Score matrix

*[PENDING — insert screenshot or table from Visual Analytics dashboard]*

### 6.3 Key finding: Grad-CAM and ground-truth part overlap

Across both preliminary (15-sample) runs, Grad-CAM consistently showed the highest part-overlap ratio and lowest clutter-leakage ratio of the three XAI methods on FunnyBirds — i.e., not only did it score best on the internal faithfulness metrics, it also most reliably pointed at the bird's actual anatomical features rather than background distractor objects. This is the central empirical claim of the project: faithfulness-metric performance and ground-truth correctness aligned for Grad-CAM, which is not guaranteed by construction and is worth highlighting as the platform's primary finding.

*[PENDING: update with final 60-sample numbers and a bar chart from the dashboard]*

## 7. Limitations

- **Tail excluded from ground truth.** Confirmed absent as a distinct color across 45 sampled images; likely camera occlusion rather than uncolored geometry. Ground-truth evaluation covers beak, eye, foot, and wing only.
- **FunnyBirds test set is small** (500 images), so per-epoch test accuracy showed more run-to-run variance than CIFAR-10's 10,000-image test set.
- **SHAP explanations use a 64×64 downsampled resolution** internally for stability reasons (§5); this is a documented approximation, not a methodological choice motivated by explanation quality.
- **Explainability Score composite weighting is uniform** (five metrics averaged equally) — a domain expert might reasonably weight faithfulness above complexity, for example. The formula is fully documented so this can be adjusted.

## 8. Future Work

- **Intervention-based ground truth.** FunnyBirds ships an official evaluation mechanism (`test_interventions/`) that measures part importance via actual 3D scene re-rendering (removing/replacing parts and observing prediction change) rather than static pixel-color matching. This is more rigorous than the pixel-overlap metric used here and would be a natural extension.
- **Larger XAI sample counts** for even tighter confidence intervals on the faithfulness/stability metrics.
- **Additional XAI methods** (e.g. Integrated Gradients, already available via Captum in the environment) could be added to the same unified interface with minimal changes.

## 9. Conclusion

*[To finalize once final results are in]*

---

## Appendix: Cited work for dataset/methodology justification

- Hesse et al., "FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods," ICCV 2023. (arXiv:2308.06248)
- Saliency-Bench (2023, arXiv:2310.08537)
- ExplainTS (2026, Frontiers in AI)
- BEExAI (2025, Springer)
- XAI-Units (2025, ACM FAccT)
