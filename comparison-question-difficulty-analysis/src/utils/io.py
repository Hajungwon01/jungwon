import json
import os
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return a list of dicts.
    Each line in the file must be a valid JSON object.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    """
    Save a list of dicts to a JSONL file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    """
    Save a dict to a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists.
    """
    os.makedirs(path, exist_ok=True)
