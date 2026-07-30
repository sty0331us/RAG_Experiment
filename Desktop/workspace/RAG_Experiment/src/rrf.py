"""Reciprocal Rank Fusion helpers shared across RAG frameworks."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[T]],
    *,
    top_k: int,
    k_rrf: int = 60,
    key_fn=None,
) -> List[Tuple[T, float]]:
    """
    Fuse multiple ranked result lists with Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : sequences of items already ordered best → worst
    top_k        : number of fused results to return
    k_rrf        : RRF constant (default 60)
    key_fn       : optional callable to derive a stable merge key from an item.
                   When omitted, the item itself is used as the key (must be hashable).

    Returns
    -------
    List of (item, rrf_score) sorted by descending score.
    When the same key appears in multiple lists, the first-seen item is kept.
    """
    scores: Dict[object, float] = {}
    items: Dict[object, T] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = key_fn(item) if key_fn is not None else item
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank + 1)
            items.setdefault(key, item)

    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [(items[key], scores[key]) for key in ordered]


def fuse_text_rankings(
    ranked_text_lists: Sequence[Iterable[str]],
    *,
    top_k: int,
    k_rrf: int = 60,
    key_len: int = 80,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper for frameworks that only have free-text chunks.

    Returns a list of {"rank", "score", "text"} dicts.
    """
    lists = [list(texts) for texts in ranked_text_lists]
    fused = reciprocal_rank_fusion(
        lists,
        top_k=top_k,
        k_rrf=k_rrf,
        key_fn=lambda text: text[:key_len],
    )
    return [
        {"rank": i + 1, "score": score, "text": text}
        for i, (text, score) in enumerate(fused)
    ]
