from typing import Dict, Any
from collections import Counter

from src.evaluation.path_labeler import label_all_paths


def compute_path_distribution(
    grouped_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute distribution of C/W reasoning paths.

    Args:
        grouped_data:
            {
              group_id: {
                "multi": {...},
                "subs": [{...}, {...}]
              }
            }

    Returns:
        {
          "counts": { "C/C/C": n, "C/W/W": n, ... },
          "ratios": { "C/C/C": r, "C/W/W": r, ... },
          "total": N
        }
    """
    path_buckets = label_all_paths(grouped_data)

    counts = Counter()
    for path, groups in path_buckets.items():
        counts[path] = len(groups)

    total = sum(counts.values())

    ratios = {
        path: (count / total if total > 0 else 0.0)
        for path, count in counts.items()
    }

    return {
        "counts": dict(counts),
        "ratios": ratios,
        "total": total,
    }


def print_path_distribution(dist: Dict[str, Any]) -> None:
    """
    Pretty-print path distribution.
    """
    counts = dist["counts"]
    ratios = dist["ratios"]
    total = dist["total"]

    print(f"Total groups: {total}")
    print("-" * 40)

    for path in sorted(counts.keys()):
        c = counts[path]
        r = ratios[path] * 100
        print(f"{path:7s} : {c:5d} ({r:6.2f}%)")
