# XAIBench — Explainable AI Benchmark Platform

A benchmark platform that trains multiple image classification models, applies multiple XAI (explainable AI) methods to each, and scores every combination on faithfulness, stability, complexity, runtime — and, uniquely, **ground-truth correctness** using a synthetic dataset where the "right answer" is actually known.

Most XAI demos show you one method on one model and let you eyeball whether the heatmap looks reasonable. XAIBench instead asks: *of these three explanation methods, which one is actually telling the truth?*

## What makes this different

The [FunnyBirds dataset](https://arxiv.org/abs/2308.06248) ships each image with a matching ground-truth part-map — a rendering that tells you exactly which pixels belong to the beak, eye, foot, and wing. This project uses that to build a metric (`funnybirds_ground_truth.py`) that checks whether an explanation's most "important" pixels actually land on a real anatomical part, or on background clutter. The canonical part colors used by this metric were verified empirically (not assumed) by scanning pixel distributions across 45 sample images spanning 15 classes — see `inspect_part_colors.py` / `scan_all_part_colors.py`.

## Architecture

```
Dataset Module → Model Module → XAI Module → Benchmark Engine → Visual Analytics
                                      ↑
                        FunnyBirds Ground-Truth Module
```

| Module | File | What it does |
|---|---|---|
| Dataset | `dataset_module.py` | Loads CIFAR-10 and FunnyBirds with matching preprocessing |
| Model | `model_module.py` | Unified interface for SimpleCNN, ResNet18, EfficientNet-B0 |
| XAI | `xai_module.py` | Grad-CAM, SHAP, LIME — normalized to a common `(H, W)` output format |
| Ground Truth | `funnybirds_ground_truth.py` | Scores explanations against real part locations |
| Benchmark Engine | `benchmark_module.py` | Ties everything together, computes all metrics, writes results CSVs |
| Training | `train_full.py` | Full training runs with checkpointing and resumability |
| Visual Analytics | `visual_analytics.html` | Interactive dashboard over the results CSVs |

## Setup

```bash
python -m venv xaibench-env
# Windows: xaibench-env\Scripts\activate
# Mac/Linux: source xaibench-env/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

See `requirements.txt` for why the CUDA install step is separate. Tested on Python 3.11.

## Usage

```bash
cd src

# 1. Verify the environment
python dataset_module.py       # downloads CIFAR-10, checks for FunnyBirds
python model_module.py         # smoke-tests all three model architectures
python xai_module.py           # smoke-tests all three XAI methods

# 2. Train (resumable — safe to interrupt and rerun)
python train_full.py --datasets cifar10 funnybirds

# 3. Benchmark trained checkpoints
python benchmark_module.py --dataset cifar10 --models resnet18 efficientnet simplecnn
python benchmark_module.py --dataset funnybirds --models resnet18 efficientnet simplecnn

# 4. Open visual_analytics.html in a browser, or use its "Load CSV" button
#    to view results/benchmark_results_*.csv
```

FunnyBirds itself isn't included in this repo (see `.gitignore`) — download it via the [official framework](https://github.com/visinf/funnybirds-framework) into `data/FunnyBirds/`.

## Results

Full results, charts, and the composite Explainability Score matrix are in `visual_analytics.html` — open it directly, no server needed. Key finding: **Grad-CAM consistently shows the highest ground-truth part-overlap and lowest clutter-leakage** of the three methods on FunnyBirds, meaning it isn't just scoring well on self-referential faithfulness metrics — it's actually pointing at the correct anatomical features more reliably than SHAP or LIME.

See `XAIBench_Report_Draft.md` for the full write-up including methodology, limitations, and the engineering issues found along the way.

## Notable engineering issues found & fixed

A few non-obvious bugs came up during development, documented here since they're the kind of thing that silently corrupts results if missed:

- **Device corruption**: the XAI methods used to force the model back onto CPU as a side effect on every call, since the device argument was never actually passed through from the Benchmark Engine. Fixed by having each method read the model's real device instead of assuming one.
- **LIME's `top_labels` default silently overriding an explicit label request** — caused `'Label not in explanation'` failures that looked like a model-quality issue but were actually a LIME API default. Fixed with `top_labels=None`.
- **SHAP hanging indefinitely at full 224×224 resolution** on this project's GPU/torch combination. Fixed by downsampling to 64×64 before running SHAP specifically (output is resized back afterward).

## License / attribution

FunnyBirds: Hesse et al., ICCV 2023. CIFAR-10: Krizhevsky, 2009. This project's code is original.
