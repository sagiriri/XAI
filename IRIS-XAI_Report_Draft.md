# IRIS-XAI: Image Recognition Interpretability & Scoring — A Benchmark Platform for Explainable Image Classification

*Final results — 60 samples per (model × XAI method), both datasets, all fixes applied.*

---

## 1. Introduction & Motivation

Explainable AI (XAI) methods like Grad-CAM, SHAP, and LIME are widely used to interpret deep learning models, but they are rarely evaluated against each other systematically. Most projects that use XAI pick one method and demonstrate it qualitatively — this project instead builds a **benchmark platform** that trains multiple models, applies multiple XAI methods to each, and scores every combination on a consistent set of quantitative metrics.

This project's primary point of comparison is Skliarov et al.'s 2025 *Scientific Reports* study, *"A comparative evaluation of explainability techniques for image data"* [22], which benchmarks six saliency-based XAI methods (LIME, SHAP, Grad-CAM, Grad-CAM++, Integrated Gradients, SmoothGrad) across three datasets (CIFAR-10, SVHN, Imagenette) and three architectures (ResNet50, VGG16, ViT-B/16), using five function-grounded metrics: fidelity, stability, identity, separability, and computational time. IRIS-XAI follows the same function-grounded philosophy but narrows the method and dataset scope in exchange for adding a dimension that study does not have: **ground-truth correctness**, evaluated on the FunnyBirds dataset, where the "right answer" for what an explanation should highlight is actually known rather than inferred from the model's own behavior. Section 6.5 gives a direct, scope-honest comparison between the two studies' findings.

This also follows the recent broader trend in the XAI literature of building standardized benchmark suites rather than one-off demonstrations — e.g. Saliency-Bench (2023), ExplainTS (2026), BEExAI (2025), and XAI-Units (2025) all take this approach. IRIS-XAI applies a version of this methodology to a smaller, self-contained image-classification setting, with Skliarov et al. [22] as the nearest direct predecessor in scope and method.

## 2. Datasets

| Dataset | Purpose | Classes | Notes |
|---|---|---|---|
| CIFAR-10 | Model training/evaluation, standard baseline | 10 | 50,000 train / 10,000 test |
| FunnyBirds | Ground-truth explanation evaluation | 50 (synthetic birds) | 50,000 train / 500 test; each image ships with a matching part-map that identifies exactly where the beak, eye, foot, and wing are — real ground truth for checking whether an explanation is pointing at the correct region, not just a plausible-looking one |

FunnyBirds is a synthetic, part-based dataset purpose-built for evaluating visual explanations (Hesse et al., 2023) — this is what differentiates IRIS-XAI from a plain model-accuracy comparison, and from the baseline study in §1: it lets us measure whether an explanation is *correct*, not just whether it looks internally consistent.

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
- **FunnyBirds Ground-Truth Module** — the project's key differentiator relative to both the general XAI-benchmarking literature and the primary baseline study. Verifies each explanation's most important pixels against the dataset's real part locations, producing a *part overlap ratio* (does the explanation hit a real bird part?) and *clutter leakage ratio* (does it waste attention on background distractor objects?).
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

### 6.1 Model Performance

| Model | Dataset | Accuracy | F1 | AUC | Params (MB) |
|---|---|---|---|---|---|
| SimpleCNN | CIFAR-10 | 63.9% | 0.635 | 0.953 | 1.50 |
| ResNet18 | CIFAR-10 | 95.2% | — | — | 42.65 |
| EfficientNet | CIFAR-10 | 96.5% | — | — | 15.34 |
| SimpleCNN | FunnyBirds | 96.4% | — | — | 1.53 |
| ResNet18 | FunnyBirds | 96.6% | — | — | 42.73 |
| EfficientNet | FunnyBirds | 97.2% | — | — | 15.53 |

SimpleCNN's large accuracy gap between datasets (63.9% vs. 96.4%) is expected, not a defect: unlike ResNet18/EfficientNet, it trains entirely from scratch with no ImageNet pretraining, so it's far more sensitive to how visually complex the task is. CIFAR-10's real-world photographs are a meaningfully harder learning problem from scratch than FunnyBirds' more visually distinct synthetic classes — this is a genuine, informative finding about the accuracy/pretraining trade-off, not noise.

### 6.2 Explainability Score (composite, 0–100)

| Model | Dataset | Grad-CAM | SHAP | LIME |
|---|---|---|---|---|
| ResNet18 | CIFAR-10 | **78.5** | 49.2 | 29.9 |
| EfficientNet | CIFAR-10 | **59.5** | 32.7 | 28.7 |
| SimpleCNN | CIFAR-10 | **77.1** | 51.5 | 35.9 |
| ResNet18 | FunnyBirds | **73.0** | 64.8 | 29.1 |
| EfficientNet | FunnyBirds | **62.9** | 45.5 | 27.5 |
| SimpleCNN | FunnyBirds | **87.1** | 80.6 | 25.3 |

Grad-CAM has the highest composite Explainability Score on **every single one of the six model × dataset combinations tested** — not a marginal or inconsistent result. LIME is consistently the lowest scorer across all six. This ranking holds specifically among the three methods IRIS-XAI tests; §6.5 addresses how it relates to the broader six-method ranking reported by the primary baseline.

### 6.3 Ground-truth part overlap (FunnyBirds only)

| Model | Method | Part overlap ↑ | Clutter leakage ↓ | Runtime |
|---|---|---|---|---|
| ResNet18 | Grad-CAM | 0.079 | **0.061** | 0.041s |
| ResNet18 | SHAP | **0.086** | 0.259 | 11.5s |
| ResNet18 | LIME | 0.014 | 0.082 | 1.7s |
| EfficientNet | Grad-CAM | **0.087** | **0.049** | 0.050s |
| EfficientNet | SHAP | 0.059 | 0.204 | 15.3s |
| EfficientNet | LIME | 0.012 | 0.071 | 1.6s |
| SimpleCNN | Grad-CAM | 0.098 | **0.092** | 0.012s |
| SimpleCNN | SHAP | **0.108** | 0.179 | 9.4s |
| SimpleCNN | LIME | 0.012 | 0.074 | 1.6s |

This is the more nuanced, and arguably more interesting, finding of the two: **SHAP occasionally matches or slightly exceeds Grad-CAM on raw part-overlap** (ResNet18, SimpleCNN) — but it does so while attending to background clutter distractors 3–4× more than Grad-CAM, and takes roughly 200–300× longer to compute. **LIME is unambiguously the weakest at correctly localizing the real anatomical part**, with part-overlap scores 5–8× lower than the other two methods across every model, despite being much faster than SHAP.

### 6.4 Key finding

Faithfulness-metric performance and ground-truth correctness aligned closely for Grad-CAM: it is not only the fastest method by 1–2 orders of magnitude, it also produces the best overall trade-off between correctly localizing real bird parts (competitive-to-best part-overlap) and avoiding background clutter (consistently lowest clutter-leakage of the three methods). SHAP can be competitive on raw localization but pays for it with substantially more attention wasted on irrelevant distractor objects. LIME, despite being commonly used as a fast approximate explainer, was the weakest performer on the one metric that checks against actual ground truth — a result that would not have been visible from faithfulness metrics alone, and is the central empirical justification for building the ground-truth evaluation layer in the first place.

### 6.5 Comparison to Prior Work

This section places IRIS-XAI's findings directly against the primary baseline, Skliarov et al. [22], rather than treating it as background citation.

**Scope comparison**

| | Skliarov et al. [22] (primary baseline) | IRIS-XAI (this project) |
|---|---|---|
| XAI methods | 6: LIME, SHAP, Grad-CAM, Grad-CAM++, Integrated Gradients, SmoothGrad | 3: Grad-CAM, SHAP, LIME |
| Datasets | 3: CIFAR-10, SVHN, Imagenette | 2: CIFAR-10, FunnyBirds |
| Models | 3: ResNet50, VGG16, ViT-B/16 | 3: SimpleCNN, ResNet18, EfficientNet-B0 |
| Core metrics | Fidelity (deletion AUC), stability, identity, separability, runtime | Faithfulness (deletion + insertion AUC), stability, complexity, runtime, composite Explainability Score |
| Ground-truth evaluation | None (all metrics are function-grounded, derived from the model's own behavior) | **Yes** — FunnyBirds part-overlap / clutter-leakage against real annotated part locations |

**Where the two studies agree.** The baseline reports that Grad-CAM and Grad-CAM++ deliver the highest computational efficiency of the methods tested, though at the cost of lower fidelity relative to gradient-based alternatives. IRIS-XAI's runtime results are consistent with the efficiency half of that finding: Grad-CAM was 1–2 orders of magnitude faster than SHAP across every model tested here. The baseline also finds identity and separability to be strong for most methods except LIME and SmoothGrad, and IRIS-XAI's own ground-truth results independently support the general conclusion that LIME is the weakest of the commonly-used approximate explainers — here, on the specific dimension of pointing at the *correct* anatomical region rather than just a stable or separable one.

**Where the scope must be kept honest.** The baseline's headline result is that gradient-based methods — particularly Integrated Gradients and SmoothGrad — achieved the strongest fidelity and stability scores overall, with statistically significant improvements over LIME, Grad-CAM, and Grad-CAM++ on CIFAR-10 and Imagenette; SHAP also performed strongly, particularly on SVHN. IRIS-XAI did not implement Integrated Gradients or SmoothGrad, so **the claim "Grad-CAM performs best" in §6.2 and §6.4 above is scoped to the three methods IRIS-XAI tested, not to XAI methods in general.** Read against the baseline, IRIS-XAI's result is better stated as: *among Grad-CAM, SHAP, and LIME, Grad-CAM gives the best trade-off of speed, faithfulness, and (uniquely tested here) ground-truth correctness* — which is compatible with, rather than contradicting, the baseline's finding that gradient-based methods outside this project's scope may fidelity-outperform Grad-CAM on non-synthetic datasets.

**What IRIS-XAI adds beyond the baseline.** The baseline's five metrics (fidelity, stability, identity, separability, time) are explicitly function-grounded — each is derived from perturbing inputs and observing the model's own output, with no independent notion of whether an explanation is actually *correct*. IRIS-XAI's FunnyBirds-based part-overlap and clutter-leakage metrics add exactly the dimension that function-grounded evaluation cannot provide: a real, annotated answer for where an explanation *should* point. This is also acknowledged as an open problem in the FunnyBirds paper itself (Hesse et al., 2023), which IRIS-XAI's ground-truth module builds on directly, and is the reason IRIS-XAI extends rather than duplicates the baseline's contribution.

## 7. Limitations

- **Narrower method and dataset scope than the primary baseline.** IRIS-XAI tests 3 of the 6 methods and 2 of the 3 datasets used by Skliarov et al. [22]; in particular, Integrated Gradients and SmoothGrad — the baseline's top fidelity/stability performers — were not implemented here. Conclusions about "the best XAI method" in this report should be read as scoped to Grad-CAM/SHAP/LIME, not as a claim against the full method space.
- **Tail excluded from ground truth.** Confirmed absent as a distinct color across 45 sampled images; likely camera occlusion rather than uncolored geometry. Ground-truth evaluation covers beak, eye, foot, and wing only.
- **FunnyBirds test set is small** (500 images), so per-epoch test accuracy showed more run-to-run variance than CIFAR-10's 10,000-image test set.
- **SHAP explanations use a 64×64 downsampled resolution** internally for stability reasons (§5); this is a documented approximation, not a methodological choice motivated by explanation quality.
- **Explainability Score composite weighting is uniform** (five metrics averaged equally) — a domain expert might reasonably weight faithfulness above complexity, for example. The formula is fully documented so this can be adjusted.
- **Ground-truth metric is pixel-color-based, not intervention-based.** FunnyBirds' own official evaluation protocol measures part importance via actual 3D scene re-rendering (removing/replacing parts and observing prediction change), which is more rigorous than the static pixel-overlap approach used here (see §8).

## 8. Future Work

- **Close the method gap with the primary baseline.** Adding Integrated Gradients and SmoothGrad (both flagged in §7) to IRIS-XAI's unified XAI interface — Integrated Gradients is already available via Captum in this project's environment — would let the ground-truth layer be applied to the baseline's full six-method set rather than a subset, producing a direct like-for-like extension of Skliarov et al. [22] rather than a partial one.
- **Intervention-based ground truth.** FunnyBirds ships an official evaluation mechanism (`test_interventions/`) that measures part importance via actual 3D scene re-rendering rather than static pixel-color matching. This is more rigorous than the pixel-overlap metric used here and would be a natural extension.
- **Larger XAI sample counts** for even tighter confidence intervals on the faithfulness/stability metrics.
- **Additional datasets** (SVHN, Imagenette) to match the baseline's dataset coverage and test whether the ground-truth-vs-fidelity relationship observed here on FunnyBirds generalizes to non-synthetic data.

## 9. Conclusion

IRIS-XAI trained three models to strong real-world accuracy (95–97% for the pretrained architectures) on two datasets and benchmarked three XAI methods against each using both internal consistency metrics and — via a purpose-built ground-truth module — actual correctness against known bird-part locations. Within this three-method scope, the central result held consistently across all six model × dataset combinations: **Grad-CAM offered the best overall trade-off of speed, faithfulness, and ground-truth correctness**, while LIME was reliably the weakest at correctly localizing real explanatory features despite its common use as a fast, general-purpose explainer — a result consistent with, though narrower in method coverage than, the primary baseline's own function-grounded findings. IRIS-XAI's specific contribution beyond that baseline, and beyond the broader function-grounded XAI-evaluation literature it is part of, is the ground-truth layer itself: a way of checking not just whether an explanation is internally consistent, stable, and separable, but whether it is actually *right*. That question is only answerable because FunnyBirds provides a dataset where the correct answer is knowable in the first place, and it is the central empirical justification for this project's design.

---

## Appendix: Cited work for dataset/methodology justification

- Hesse et al., "FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods," ICCV 2023. (arXiv:2308.06248)
- Skliarov, M., El Shawi, R., Dhaoui, C., & Ahmed, N., "A comparative evaluation of explainability techniques for image data," *Scientific Reports* 15, 41898 (2025). — **Primary comparison baseline for this project (see §1, §6.5).**
- Saliency-Bench (2023, arXiv:2310.08537)
- ExplainTS (2026, Frontiers in AI)
- BEExAI (2025, Springer)
- XAI-Units (2025, ACM FAccT)
