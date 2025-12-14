from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.utils.io import load_jsonl


def load_dataset(
    path: str,
    hop: Optional[int] = None,
    question_type: Optional[str] = None,
    role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load a JSONL dataset and optionally filter by hop count,
    question type (comparison / bridge), and role (multi / sub).

    Args:
        path: Path to JSONL file
        hop: 2 or 3
        question_type: "comparison" or "bridge"
        role: "multi" or "sub"

    Returns:
        List of dataset samples
    """
    data = load_jsonl(path)

    filtered = []
    for item in data:
        meta = item.get("metadata", {})

        if hop is not None and meta.get("hop") != hop:
            continue

        if question_type is not None and meta.get("question_type") != question_type:
            continue

        if role is not None:
            qtype = item.get("qtype", "")
            if role == "multi" and qtype != "multi":
                continue
            if role == "sub" and not qtype.startswith("sub"):
                continue

        filtered.append(item)

    return filtered


def group_by_group_id(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group dataset items by group_id.
    """
    groups = defaultdict(list)
    for item in data:
        groups[item["group_id"]].append(item)
    return groups


def load_grouped_dataset(
    path: str,
    hop: Optional[int] = None,
    question_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load dataset and return grouped data by group_id.
    Each group contains multi-question and sub-questions.

    Returns:
        {
          group_id: {
            "multi": {...},
            "subs": [{...}, {...}]
          }
        }
    """
    data = load_dataset(path, hop=hop, question_type=question_type)
    grouped = group_by_group_id(data)

    output = {}

    for group_id, items in grouped.items():
        multi = None
        subs = []

        for item in items:
            if item["qtype"] == "multi":
                multi = item
            else:
                subs.append(item)

        if multi is None:
            # Skip incomplete groups
            continue

        output[group_id] = {
            "multi": multi,
            "subs": sorted(subs, key=lambda x: x["qtype"])
        }

    return output
