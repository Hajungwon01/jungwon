import re
from typing import List, Dict, Any
from collections import Counter


def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Standard SQuAD-style normalization.
    """
    if s is None:
        return ""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return re.sub(r"[^\w\s]", "", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """
    Exact Match (EM) score.
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    predictions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Compute EM and F1 for each prediction record.

    Input record must contain:
        - prediction
        - golden_answers
    """
    results = []

    for item in predictions:
        pred = item.get("prediction", "")
        gold = item.get("golden_answers", "")

        em = exact_match_score(pred, gold)
        f1 = f1_score(pred, gold)

        out = item.copy()
        out["em"] = em
        out["f1"] = f1

        results.append(out)

    return results


def aggregate_metrics(
    predictions: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Aggregate EM and F1 over all predictions.
    """
    if len(predictions) == 0:
        return {"em": 0.0, "f1": 0.0}

    ems = [p["em"] for p in predictions]
    f1s = [p["f1"] for p in predictions]

    return {
        "em": sum(ems) / len(ems),
        "f1": sum(f1s) / len(f1s),
    }
