from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.evaluation.metrics import compute_metrics, aggregate_metrics


def _get_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe getter for metadata fields.
    """
    return item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}


def _role_from_qtype(qtype: str) -> str:
    """
    Map qtype to role: multi vs sub.
    """
    if qtype == "multi":
        return "multi"
    if qtype.startswith("sub"):
        return "sub"
    return "other"


def score_predictions(
    predictions: List[Dict[str, Any]],
    group_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute per-sample EM/F1, then aggregate metrics across multiple groupings.

    Args:
        predictions: list of prediction records.
                     Each record should have:
                     - prediction
                     - golden_answers
                     - qtype
                     - (optional) metadata: {hop, question_type, ...}
        group_keys: list of grouping keys to compute breakdown.
                    Supported keys:
                    - "model"
                    - "qtype"
                    - "role"          (multi/sub)
                    - "hop"           (from metadata)
                    - "question_type" (from metadata; comparison/bridge)

    Returns:
        {
          "overall": {"em": ..., "f1": ...},
          "breakdown": {
              "<group_key>": {
                  "<value>": {"em": ..., "f1": ..., "n": ...},
                  ...
              },
              ...
          },
          "scored": [ ... per-sample with em/f1 ... ]
        }
    """
    if group_keys is None:
        group_keys = ["model", "qtype", "role", "hop", "question_type"]

    scored = compute_metrics(predictions)
    overall = aggregate_metrics(scored)

    breakdown = {}

    for key in group_keys:
        buckets = defaultdict(list)

        for item in scored:
            meta = _get_meta(item)

            if key == "model":
                g = item.get("model", "unknown")
            elif key == "qtype":
                g = item.get("qtype", "unknown")
            elif key == "role":
                g = _role_from_qtype(item.get("qtype", ""))
            elif key == "hop":
                g = meta.get("hop", "unknown")
            elif key == "question_type":
                g = meta.get("question_type", "unknown")
            else:
                # generic fallback: try in item first then metadata
                g = item.get(key, meta.get(key, "unknown"))

            buckets[str(g)].append(item)

        breakdown[key] = {
            g: {**aggregate_metrics(items), "n": len(items)}
            for g, items in buckets.items()
        }

    return {
        "overall": overall,
        "breakdown": breakdown,
        "scored": scored,
    }


def score_by_multi_sub(
    predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute metrics for multi vs subs.
    """
    scored = compute_metrics(predictions)

    multi = [x for x in scored if x.get("qtype") == "multi"]
    subs = [x for x in scored if str(x.get("qtype", "")).startswith("sub")]

    return {
        "multi": {**aggregate_metrics(multi), "n": len(multi)},
        "sub": {**aggregate_metrics(subs), "n": len(subs)},
        "overall": {**aggregate_metrics(scored), "n": len(scored)},
    }
