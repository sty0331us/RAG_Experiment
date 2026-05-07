"""Result visualisation utilities."""
from __future__ import annotations

import math
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

_FRAMEWORK_COLORS = {
    "llamaindex": "#4C72B0",
    "langchain":  "#DD8452",
    "langgraph":  "#55A868",
}


def _summary_to_df(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, metrics in summary.items():
        framework, method = key.split("/", 1)
        rows.append({"framework": framework, "method": method, **metrics})
    return pd.DataFrame(rows)


def plot_latency(summary: Dict[str, Any], out_dir: Path) -> None:
    """Per-framework stacked bar: retrieval + generation time."""
    df = _summary_to_df(summary)
    frameworks = df["framework"].unique().tolist()
    n = len(frameworks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, fw in zip(axes, frameworks):
        sub = df[df["framework"] == fw].sort_values("total_time_s")
        if sub.empty:
            ax.set_visible(False)
            continue
        bar_data = sub[["method", "retrieval_time_s", "gen_time_s"]].set_index("method")
        color = _FRAMEWORK_COLORS.get(fw, None)
        bar_data.plot(kind="bar", stacked=True, ax=ax,
                      color=["#aec6e8", color] if color else None,
                      legend=(ax is axes[-1]))
        ax.set_title(f"Latency – {fw.upper()}", fontweight="bold")
        ax.set_xlabel("Similarity Method")
        ax.set_ylabel("Seconds")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = out_dir / "latency_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


def plot_context_relevance(summary: Dict[str, Any], out_dir: Path) -> None:
    """Grouped bar: avg context relevance per method, coloured by framework."""
    df = _summary_to_df(summary)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = df.pivot(index="method", columns="framework", values="avg_ctx_relevance")

    colors = [_FRAMEWORK_COLORS.get(c, None) for c in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.7)

    ax.set_title("Average Context Relevance per Method & Framework", fontweight="bold")
    ax.set_xlabel("Similarity Method")
    ax.set_ylabel("Avg Cosine Similarity")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Framework")
    plt.tight_layout()
    path = out_dir / "context_relevance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


def plot_framework_comparison(summary: Dict[str, Any], out_dir: Path) -> None:
    """Side-by-side bar comparing frameworks on total_time_s and avg_ctx_relevance."""
    df = _summary_to_df(summary)
    metrics = [
        ("avg_ctx_relevance", "Avg Context Relevance", "%.3f"),
        ("total_time_s",      "Avg Total Latency (s)", "%.2f"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (metric, label, fmt) in zip(axes, metrics):
        pivot = df.pivot(index="method", columns="framework", values=metric)
        colors = [_FRAMEWORK_COLORS.get(c, None) for c in pivot.columns]
        pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.7)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Similarity Method")
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Framework")

    plt.suptitle("Framework Comparison: LlamaIndex vs LangChain vs LangGraph",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = out_dir / "framework_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_heatmap(summary: Dict[str, Any], metric: str, out_dir: Path) -> None:
    df = _summary_to_df(summary)
    pivot = df.pivot(index="method", columns="framework", values=metric)
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 3), 5))
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
        print("\n" + tabulate(df, headers="keys", tablefmt="rounded_outline",
                              floatfmt=".4f", showindex=False))
    except ImportError:
        print(df.to_string(index=False))


def generate_all_plots(summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    if not summary:
        logger.warning("No results to visualise – skipping plots.")
        return
    plot_latency(summary, out_dir)
    plot_context_relevance(summary, out_dir)
    plot_framework_comparison(summary, out_dir)
    plot_heatmap(summary, "avg_ctx_relevance", out_dir)
    plot_heatmap(summary, "total_time_s", out_dir)
    save_summary_table(summary, out_dir)
