from __future__ import annotations

import numpy as np
import pandas as pd

from .profiler import format_query, profile_stream
from .sandbox import SandboxEvaluator


def make_triplets(stream: pd.Series, task_category: str, tool_descriptions: dict[str, str],
                  evaluator: SandboxEvaluator, labels: np.ndarray | None = None) -> list[dict]:
    scores = evaluator.evaluate(np.asarray(stream, dtype=float), task_category, labels)
    if len(scores) < 2 or max(scores.values()) <= 0.0:
        return []
    winner = max(scores, key=scores.__getitem__)
    query = format_query(profile_stream(stream), task_category)
    return [{
        "query":           query,
        "positive_method": winner,
        "negative_method": tool,
        "positive_key":    tool_descriptions[winner],
        "negative_key":    tool_descriptions[tool],
        "delta_s":         round(scores[winner] - scores[tool], 6),
    } for tool in scores if tool != winner]
