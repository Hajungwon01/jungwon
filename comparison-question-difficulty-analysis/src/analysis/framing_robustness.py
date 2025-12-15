"""
Framing Robustness Analysis

This module analyzes how stable model predictions and reasoning paths are
when only the framing (direction or wording) of comparison questions changes.

Model inference scripts are intentionally excluded.
This module operates on pre-generated prediction results.
"""

from typing import List, Dict, Any
from collections import defaultdict, Counter


def _is_correct(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False
    return pred.strip().lower() == gold.strip().lower()


def analyze_framing_robustness(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze robustness across different framings of the same comparison question.

    Each item in results must contain:
        - group_id        : str
        - framing_id      : str
        - prediction      : str
        - golden_answers  : str
        - path_label      : str   (e.g., C/W/W)

    Returns:
        {
          "counts": {
              "all_correct": int,
              "partially_correct": int,
              "all_wrong": int
          },
          "ratios": { ... },
          "path_transitions": { "C/W/W -> W/W/W": n, ... },
          "unstable_examples": [ {...}, ... ]
        }
    """
    grouped = defaultdict(list)
    for item in results:
        grouped[item["group_id"]].append(item)

    correctness_counter = Counter()
    path_transition_counter = Counter()
    unstable_examples = []

    for group_id, items in grouped.items():
        correctness = [
            _is_correct(x["prediction"], x["golden_answers"])
            for x in items
        ]

        if all(correctness):
            correctness_counter["all_correct"] += 1
        elif any(correctness):
            correctness_counter["partially_correct"] += 1
            unstable_examples.append(items)
        else:
            correctness_counter["all_wrong"] += 1

        # path transition analysis
        base_path = items[0]["path_label"]
        for x in items[1:]:
            if x["path_label"] != base_path:
                key = f"{base_path} -> {x['path_label']}"
                path_transition_counter[key] += 1

    total = sum(correctness_counter.values())
    ratios = {
        k: (v / total if total > 0 else 0.0)
        for k, v in correctness_counter.items()
    }

    return {
        "counts": dict(correctness_counter),
        "ratios": ratios,
        "path_transitions": dict(path_transition_counter),
        "unstable_examples": unstable_examples
    }


def summarize_framing_robustness(result: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of framing robustness analysis.
    """
    lines = []
    lines.append("Framing Robustness Summary")
    lines.append("-" * 40)

    for k, v in result["counts"].items():
        ratio = result["ratios"].get(k, 0.0) * 100
        lines.append(f"{k:20s}: {v:5d} ({ratio:6.2f}%)")

    if result["path_transitions"]:
        lines.append("\nPath Transitions")
        lines.append("-" * 40)
        for k, v in result["path_transitions"].items():
            lines.append(f"{k:20s}: {v}")

    return "\n".join(lines)
