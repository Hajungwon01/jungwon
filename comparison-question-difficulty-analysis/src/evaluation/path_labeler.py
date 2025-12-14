from typing import Dict, List, Any
from collections import defaultdict


def normalize_answer(ans: str) -> str:
    """
    Normalize answer string for exact match.
    """
    if ans is None:
        return ""
    return ans.strip().lower()


def is_correct(pred: str, gold: str) -> bool:
    """
    Exact match correctness check.
    """
    return normalize_answer(pred) == normalize_answer(gold)


def label_group_path(group: Dict[str, Any]) -> str:
    """
    Assign C/W path label to a single group.

    Input format:
    {
        "multi": {...},
        "subs": [{...}, {...}]
    }

    Output:
        Path label string (e.g., "C/W/W")
    """
    labels = []

    # 1. Multi question
    multi = group["multi"]
    labels.append(
        "C" if is_correct(multi["prediction"], multi["golden_answers"]) else "W"
    )

    # 2. Sub-questions (ordered)
    for sub in group["subs"]:
        labels.append(
            "C" if is_correct(sub["prediction"], sub["golden_answers"]) else "W"
        )

    return "/".join(labels)


def label_all_paths(
    grouped_data: Dict[str, Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Label all groups and categorize them by path pattern.

    Returns:
        {
            "C/C/C": [group1, group2, ...],
            "C/W/W": [...],
            ...
        }
    """
    path_buckets = defaultdict(list)

    for group_id, group in grouped_data.items():
        path = label_group_path(group)
        path_buckets[path].append({
            "group_id": group_id,
            "path": path,
            "multi": group["multi"],
            "subs": group["subs"],
        })

    return path_buckets
