"""Markdown comparison report writer for experiment summaries."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger


def _summary_to_df(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, metrics in summary.items():
        framework, method = key.split("/", 1)
        rows.append({"framework": framework, "method": method, **metrics})
    return pd.DataFrame(rows)


def write_markdown_report(summary: Dict[str, Any], out_path: Path) -> Path:
    """
    Write a human-readable Markdown report comparing frameworks × methods.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary:
        out_path.write_text("# RAG Experiment Report\n\nNo results available.\n", encoding="utf-8")
        logger.warning(f"Empty summary – wrote stub report to {out_path}")
        return out_path

    df = _summary_to_df(summary)
    df = df.sort_values(["framework", "avg_ctx_relevance"], ascending=[True, False])

    lines = [
        "# RAG Experiment Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary table",
        "",
        "| Framework | Method | Total latency (s) | Retrieval (s) | Generation (s) | Context relevance | Source diversity | Answer length | Queries |",
        "|-----------|--------|------------------:|--------------:|---------------:|------------------:|-----------------:|--------------:|--------:|",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {row['framework']} | {row['method']} "
            f"| {row['total_time_s']:.3f} "
            f"| {row['retrieval_time_s']:.3f} "
            f"| {row['gen_time_s']:.3f} "
            f"| {row['avg_ctx_relevance']:.4f} "
            f"| {row.get('source_diversity', 0):.3f} "
            f"| {row['answer_length']:.0f} "
            f"| {int(row['n_queries'])} |"
        )

    lines.extend(["", "## Best by metric", ""])

    best_rel = df.loc[df["avg_ctx_relevance"].idxmax()]
    best_lat = df.loc[df["total_time_s"].idxmin()]
    lines.append(
        f"- **Highest context relevance**: `{best_rel['framework']}/{best_rel['method']}` "
        f"({best_rel['avg_ctx_relevance']:.4f})"
    )
    lines.append(
        f"- **Lowest total latency**: `{best_lat['framework']}/{best_lat['method']}` "
        f"({best_lat['total_time_s']:.3f}s)"
    )

    lines.extend(["", "## Per-framework averages", ""])
    fw_avg = (
        df.groupby("framework")[["total_time_s", "avg_ctx_relevance"]]
        .mean()
        .sort_values("avg_ctx_relevance", ascending=False)
    )
    lines.append("| Framework | Avg latency (s) | Avg context relevance |")
    lines.append("|-----------|----------------:|----------------------:|")
    for fw, row in fw_avg.iterrows():
        lines.append(f"| {fw} | {row['total_time_s']:.3f} | {row['avg_ctx_relevance']:.4f} |")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved markdown report: {out_path}")
    return out_path
