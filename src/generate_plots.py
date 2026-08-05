"""
generate_plots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULT_DIR, RANKING_METHODS, FEATURE_SIZES, TARGETS

PLOT_DIR = RESULT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = plt.cm.tab10(np.linspace(0, 1, len(RANKING_METHODS)))
METHOD_COLOR = dict(zip(RANKING_METHODS, COLORS))


def load_summary(target):
    path = RESULT_DIR / target / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------
# Plot 1 & 2: Accuracy / F1 vs Top-K, one line per ranking method, per target
# --------------------------------------------------------------------------

def plot_metric_vs_topk(df, target, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method in RANKING_METHODS:
        sub = df[df["ranking_method"] == method].sort_values("top_features")
        if sub.empty:
            continue
        ax.plot(
            sub["top_features"],
            sub[metric],
            marker="o",
            label=method,
            color=METHOD_COLOR[method],
            linewidth=2,
        )

    ax.set_xlabel("Number of top-ranked features (top_k)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{target}: {ylabel} vs. Feature-Set Size")
    ax.set_xticks(FEATURE_SIZES)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Ranking Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


# --------------------------------------------------------------------------
# Plot 3: CV score vs Test accuracy, all experiments, colored by target
# --------------------------------------------------------------------------

def plot_cv_vs_test(all_df, filename="cv_vs_test_accuracy.png"):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    target_markers = {"Subject_ID": "o", "Task_ID": "^"}
    target_colors = {"Subject_ID": "#1f77b4", "Task_ID": "#d62728"}

    for target in TARGETS:
        sub = all_df[all_df["target"] == target]
        ax.scatter(
            sub["cv_score"],
            sub["accuracy"],
            marker=target_markers.get(target, "o"),
            color=target_colors.get(target, "gray"),
            label=target,
            alpha=0.75,
            s=50,
            edgecolor="black",
            linewidth=0.3,
        )

    lo = min(all_df["cv_score"].min(), all_df["accuracy"].min()) - 0.02
    hi = max(all_df["cv_score"].max(), all_df["accuracy"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1, label="y = x (perfect agreement)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("CV (validation) accuracy")
    ax.set_ylabel("Test accuracy")
    ax.set_title("CV Score vs. Test Accuracy (all 98 experiments)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


# --------------------------------------------------------------------------
# Plot 4: Confusion matrix heatmap for the best model per target
# --------------------------------------------------------------------------

def plot_confusion_matrix(target, ranking_method, top_k, filename):
    cm_path = (
        RESULT_DIR
        / target
        / "confusion_matrices"
        / f"{ranking_method}_Top{top_k}.csv"
    )
    if not cm_path.exists():
        print(f"WARNING: confusion matrix not found: {cm_path}")
        return

    cm_df = pd.read_csv(cm_path, index_col=0)
    cm = cm_df.values
    labels = cm_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), max(5, len(labels) * 0.4)))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{target} — Best Model\n({ranking_method}, Top {top_k})")

    # Annotate cells only if matrix is small enough to stay readable
    if len(labels) <= 15:
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6,
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


# --------------------------------------------------------------------------
# Plot 5: Heatmap summary matrix (ranking_method x top_k), colored by accuracy
# --------------------------------------------------------------------------

def plot_heatmap_summary(df, target, filename):
    pivot = df.pivot(index="ranking_method", columns="top_features", values="accuracy")
    pivot = pivot.reindex(index=RANKING_METHODS, columns=FEATURE_SIZES)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(FEATURE_SIZES)))
    ax.set_xticklabels(FEATURE_SIZES)
    ax.set_yticks(range(len(RANKING_METHODS)))
    ax.set_yticklabels(RANKING_METHODS)
    ax.set_xlabel("Top-K features")
    ax.set_ylabel("Ranking method")
    ax.set_title(f"{target}: Test Accuracy Heatmap")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < pivot.values.max() * 0.6 else "black",
                        fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Test Accuracy")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    all_dfs = []

    for target in TARGETS:
        df = load_summary(target)
        df["target"] = target
        all_dfs.append(df)

        plot_metric_vs_topk(
            df, target, metric="accuracy", ylabel="Test Accuracy",
            filename=f"{target}_accuracy_vs_topk.png",
        )
        plot_metric_vs_topk(
            df, target, metric="f1_score", ylabel="Test F1-score (macro)",
            filename=f"{target}_f1_vs_topk.png",
        )
        plot_heatmap_summary(
            df, target, filename=f"{target}_accuracy_heatmap.png",
        )

        # Best model per target = highest test accuracy row
        best_row = df.loc[df["accuracy"].idxmax()]
        plot_confusion_matrix(
            target,
            best_row["ranking_method"],
            int(best_row["top_features"]),
            filename=f"{target}_best_confusion_matrix.png",
        )
        print(
            f"Best {target} model: {best_row['ranking_method']} | "
            f"Top {int(best_row['top_features'])} | "
            f"accuracy={best_row['accuracy']:.4f} | f1={best_row['f1_score']:.4f}"
        )

    all_df = pd.concat(all_dfs, ignore_index=True)
    plot_cv_vs_test(all_df)

    print(f"\nAll plots saved to: {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()