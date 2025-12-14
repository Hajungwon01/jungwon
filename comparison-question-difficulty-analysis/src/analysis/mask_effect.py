from typing import Dict, Any

from src.analysis.path_distribution import compute_path_distribution
from src.dataset.loader import load_grouped_dataset


def analyze_mask_effect(
    original_path: str,
    masked_path: str,
    hop: int = None,
    question_type: str = None,
) -> Dict[str, Any]:
    """
    Compare reasoning path distributions between original and masked datasets.

    Args:
        original_path: path to original dataset (JSONL)
        masked_path: path to masked dataset (JSONL)
        hop: 2 or 3
        question_type: comparison / bridge

    Returns:
        {
          "original": {...},
          "masked": {...},
          "delta": {...}
        }
    """
    original_groups = load_grouped_dataset(
        original_path, hop=hop, question_type=question_type
    )
    masked_groups = load_grouped_dataset(
        masked_path, hop=hop, question_type=question_type
    )

    orig_dist = compute_path_distribution(original_groups)
    mask_dist = compute_path_distribution(masked_groups)

    delta = {}
    for path in orig_dist["ratios"]:
        delta[path] = (
            mask_dist["ratios"].get(path, 0.0)
            - orig_dist["ratios"].get(path, 0.0)
        )

    return {
        "original": orig_dist,
        "masked": mask_dist,
        "delta": delta,
    }
