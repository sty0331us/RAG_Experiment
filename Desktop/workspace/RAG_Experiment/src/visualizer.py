"""Result visualisation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["font.family"] = "AppleGothic"   # Korean font on macOS


def _summary_to_df(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, metrics in summary.items():
        framework, method = key.split("/", 1)
        rows.append({"framework": framework, "method": method, **metrics})
    return pd.DataFrame(rows)


def plot_latency(summary: Dict[str, Any], out_dir: Path) -> None:
    df = _summary_to_df(summary)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, framework in zip(axes, ["llamaindex", "langchain"]):
        sub = df[df["framework"] == framework].sort_values("total_time_s")
        if sub.empty:
            continue
        bar_data = sub[["method", "retrieval_time_s", "gen_time_s"]].set_index("method")
        bar_data.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", legend=ax == axes[1])
        ax.set_title(f"Latency – {framework.upper()}", fontweight="bold")
        ax.set_xlabel("Similarity Method")
        ax.set_ylabel("Seconds")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = out_dir / "latency_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


def plot_context_relevance(summary: Dict[str, Any], out_dir: Path) -> None:
    df = _summary_to_df(summary)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot(index="method", columns="framework", values="avg_ctx_relevance")
    pivot.plot(kind="bar", ax=ax, colormap="Paired", edgecolor="white")
    ax.set_title("Average Context Relevance (cosine sim to query)", fontweight="bold")
    ax.set_xlabel("Similarity Method")
    ax.set_ylabel("Avg Cosine Similarity")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path = out_dir / "context_relevance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


def plot_heatmap(summary: Dict[str, Any], metric: str, out_dir: Path) -> None:
    df = _summary_to_df(summary)
    pivot = df.pivot(index="method", columns="framework", values=metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title(f"Heatmap – {metric}", fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"heatmap_{metric}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


def save_summary_table(summary: Dict[str, Any], out_dir: Path) -> None:
    df = _summary_to_df(summary)
    df = df.sort_values(["framework", "avg_ctx_relevance"], ascending=[True, False])
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"Saved: {csv_path}")

    try:
        from tabulate import tabulate
        print("\n" + tabulate(df, headers="keys", tablefmt="rounded_outline", floatfmt=".4f", showindex=False))
    except ImportError:
        print(df.to_string(index=False))


def generate_all_plots(summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    if not summary:
        logger.warning("No results to visualise – skipping plots.")
        return
    plot_latency(summary, out_dir)
    plot_context_relevance(summary, out_dir)
    plot_heatmap(summary, "avg_ctx_relevance", out_dir)
    plot_heatmap(summary, "total_time_s", out_dir)
    save_summary_table(summary, out_dir)
