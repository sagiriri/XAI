"""
generate_results.py
--------------------
IRIS-XAI — final results generator.

Reads the already-completed master benchmark CSV (18 rows = 3 models x 3 XAI
methods x 2 datasets, 60 samples each) and produces every downstream artifact
needed for the report: comparison tables, the Explainability Score matrix,
the FunnyBirds ground-truth table, method rankings, and figures.

DOES NOT re-run any benchmarks, training, or XAI explanations. Read-only on
the master CSV.

Usage (from D:\\XAI\\src, with the master CSV in D:\\XAI\\results):
    python generate_results.py
    python generate_results.py --input ..\\results\\IRIS-XAI_master_results.csv --outdir ..\\results\\final
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAVE_PLOTTING = True
except ImportError:
    HAVE_PLOTTING = False

# Columns the master CSV must contain (matches visual_analytics.html's
# DEFAULT_CSV header exactly, so anything this script writes stays
# drop-in-compatible with the dashboard).
REQUIRED_COLS = [
    "dataset", "model", "xai_method", "n_xai_samples",
    "accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro_ovr",
    "avg_batch_inference_time_sec", "model_size_mb",
    "explanation_runtime_sec",
    "faithfulness_deletion_auc", "faithfulness_insertion_auc",
    "stability_cosine_sim", "max_sensitivity",
    "complexity_entropy", "complexity_entropy_normalized",
    "part_overlap_ratio", "clutter_leakage_ratio",
]

XAI_ORDER = ["gradcam", "shap", "lime"]


def load_master_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"ERROR: master CSV not found at {path}\n"
            f"This script does not create or rerun benchmarks — point --input "
            f"at your existing IRIS-XAI_master_results.csv."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: master CSV is missing expected columns: {missing}")
        print("Continuing anyway — any table/figure needing those columns will be skipped.\n")
    # normalize method names for stable grouping/ordering, keep original for display
    df["xai_method_norm"] = df["xai_method"].astype(str).str.strip().str.lower()
    return df


def min_max_norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)  # degenerate case: no spread
    return (s - lo) / (hi - lo)


def compute_explainability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite Explainability Score (0-100), per IRIS-XAI_Report_Draft.md sec 4.3:
    average of insertion AUC, (1 - deletion AUC), stability, (1 - max_sensitivity),
    (1 - complexity_entropy_normalized) -- each min-max normalized across the
    full run (all 18 rows) before averaging.
    """
    needed = [
        "faithfulness_insertion_auc", "faithfulness_deletion_auc",
        "stability_cosine_sim", "max_sensitivity", "complexity_entropy_normalized",
    ]
    if any(c not in df.columns for c in needed):
        print("Skipping Explainability Score: missing one of", needed)
        return df

    out = df.copy()
    n_ins = min_max_norm(out["faithfulness_insertion_auc"])
    n_del = 1 - min_max_norm(out["faithfulness_deletion_auc"])
    n_stab = min_max_norm(out["stability_cosine_sim"])
    n_sens = 1 - min_max_norm(out["max_sensitivity"])
    n_ent = 1 - min_max_norm(out["complexity_entropy_normalized"])

    out["explainability_score"] = ((n_ins + n_del + n_stab + n_sens + n_ent) / 5.0) * 100.0
    return out


def table_model_performance(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model) -- these metrics are model-level, identical
    across all xai_method rows for that model, so de-dupe on first occurrence."""
    cols = [c for c in [
        "dataset", "model", "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "auc_macro_ovr", "avg_batch_inference_time_sec", "model_size_mb",
    ] if c in df.columns]
    t = df[cols].drop_duplicates(subset=["dataset", "model"]).sort_values(["dataset", "model"])
    return t.reset_index(drop=True)


def table_explainability_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model x Dataset rows, Grad-CAM/SHAP/LIME columns, composite score values."""
    if "explainability_score" not in df.columns:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index=["dataset", "model"],
        columns="xai_method",
        values="explainability_score",
        aggfunc="mean",
    ).round(1)
    return pivot.reset_index()


def table_funnybirds_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in [
        "model", "xai_method", "part_overlap_ratio", "clutter_leakage_ratio",
        "explanation_runtime_sec",
    ] if c in df.columns]
    fb = df[df["dataset"].astype(str).str.lower() == "funnybirds"][cols].copy()
    if fb.empty:
        return fb
    fb = fb.sort_values(["model", "xai_method"]).reset_index(drop=True)
    return fb


def table_method_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall ranking of Grad-CAM vs SHAP vs LIME, averaged across every
    (model x dataset) combination, on: explainability_score (higher better),
    part_overlap_ratio (higher better, FunnyBirds only),
    clutter_leakage_ratio (lower better, FunnyBirds only),
    explanation_runtime_sec (lower better).
    """
    rows = []
    for method, g in df.groupby("xai_method"):
        row = {"xai_method": method}
        if "explainability_score" in g.columns:
            row["avg_explainability_score"] = g["explainability_score"].mean()
        if "explanation_runtime_sec" in g.columns:
            row["avg_runtime_sec"] = g["explanation_runtime_sec"].mean()
        fb = g[g["dataset"].astype(str).str.lower() == "funnybirds"]
        if not fb.empty:
            if "part_overlap_ratio" in fb.columns:
                row["avg_part_overlap_ratio"] = fb["part_overlap_ratio"].mean()
            if "clutter_leakage_ratio" in fb.columns:
                row["avg_clutter_leakage_ratio"] = fb["clutter_leakage_ratio"].mean()
        rows.append(row)
    rank_df = pd.DataFrame(rows)

    if "avg_explainability_score" in rank_df.columns:
        rank_df["rank_by_score"] = rank_df["avg_explainability_score"].rank(ascending=False).astype(int)
        rank_df = rank_df.sort_values("rank_by_score")
    return rank_df.reset_index(drop=True)


def make_figures(df: pd.DataFrame, outdir: Path):
    if not HAVE_PLOTTING:
        print("matplotlib/seaborn not available -- skipping figures.")
        return
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Explainability Score grouped bar chart (model x dataset on x-axis, hue=method)
    if "explainability_score" in df.columns:
        plot_df = df.copy()
        plot_df["combo"] = plot_df["model"] + " / " + plot_df["dataset"]
        order = sorted(plot_df["combo"].unique())
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=plot_df, x="combo", y="explainability_score", hue="xai_method",
            order=order, ax=ax,
        )
        ax.set_title("Explainability Score by Model / Dataset")
        ax.set_xlabel("")
        ax.set_ylabel("Explainability Score (0-100)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(figdir / "explainability_score_by_model.png", dpi=150)
        plt.close(fig)

    # 2. FunnyBirds part overlap vs clutter leakage scatter
    fb = df[df["dataset"].astype(str).str.lower() == "funnybirds"]
    if not fb.empty and {"part_overlap_ratio", "clutter_leakage_ratio"}.issubset(fb.columns):
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(
            data=fb, x="part_overlap_ratio", y="clutter_leakage_ratio",
            hue="xai_method", style="model", s=140, ax=ax,
        )
        ax.set_title("FunnyBirds: Ground-Truth Part Overlap vs. Clutter Leakage")
        ax.set_xlabel("Part overlap ratio (higher = better)")
        ax.set_ylabel("Clutter leakage ratio (lower = better)")
        plt.tight_layout()
        fig.savefig(figdir / "funnybirds_overlap_vs_leakage.png", dpi=150)
        plt.close(fig)

    # 3. Runtime comparison (log scale)
    if "explanation_runtime_sec" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x="xai_method", y="explanation_runtime_sec", hue="dataset", ax=ax)
        ax.set_yscale("log")
        ax.set_title("Explanation Runtime by XAI Method (log scale)")
        ax.set_ylabel("Seconds per explanation (log)")
        plt.tight_layout()
        fig.savefig(figdir / "runtime_by_method.png", dpi=150)
        plt.close(fig)

    print(f"Figures written to {figdir}")


def df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table; fall back to a plain
    fixed-width text block if the optional `tabulate` package isn't
    installed, so this never crashes the whole script."""
    if df.empty:
        return "_(no data)_"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_markdown_summary(outdir: Path, model_perf, expl_matrix, fb_gt, rankings):
    lines = ["# IRIS-XAI — Final Results Summary\n",
             "Auto-generated from the master benchmark CSV. No experiments were rerun.\n"]

    lines.append("## Model Performance\n")
    lines.append(df_to_md(model_perf))
    lines.append("\n")

    lines.append("## Explainability Score Matrix (0-100)\n")
    lines.append(df_to_md(expl_matrix))
    lines.append("\n")

    lines.append("## FunnyBirds Ground-Truth Comparison\n")
    lines.append(df_to_md(fb_gt))
    lines.append("\n")

    lines.append("## Overall Method Ranking (Grad-CAM vs SHAP vs LIME)\n")
    lines.append(df_to_md(rankings))
    lines.append("\n")

    out_path = outdir / "IRIS-XAI_final_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate IRIS-XAI final results from the master CSV.")
    ap.add_argument("--input", default="../results/IRIS-XAI_master_results.csv",
                     help="Path to the master results CSV (default: ../results/IRIS-XAI_master_results.csv)")
    ap.add_argument("--outdir", default="../results/final",
                     help="Directory to write tables/figures into (default: ../results/final)")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_master_csv(in_path)
    df = compute_explainability_score(df)

    model_perf = table_model_performance(df)
    expl_matrix = table_explainability_matrix(df)
    fb_gt = table_funnybirds_ground_truth(df)
    rankings = table_method_rankings(df)

    model_perf.to_csv(outdir / "model_performance.csv", index=False)
    expl_matrix.to_csv(outdir / "explainability_score_matrix.csv", index=False)
    fb_gt.to_csv(outdir / "funnybirds_ground_truth.csv", index=False)
    rankings.to_csv(outdir / "method_rankings.csv", index=False)
    df.to_csv(outdir / "master_results_with_scores.csv", index=False)

    make_figures(df, outdir)
    write_markdown_summary(outdir, model_perf, expl_matrix, fb_gt, rankings)

    print("\nDone. Wrote to:", outdir.resolve())
    print(" - model_performance.csv")
    print(" - explainability_score_matrix.csv")
    print(" - funnybirds_ground_truth.csv")
    print(" - method_rankings.csv")
    print(" - master_results_with_scores.csv")
    print(" - IRIS-XAI_final_summary.md")
    if HAVE_PLOTTING:
        print(" - figures/*.png")


if __name__ == "__main__":
    main()
