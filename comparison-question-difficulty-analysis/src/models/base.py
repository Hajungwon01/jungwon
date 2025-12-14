from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseModel(ABC):
    """
    Base class for all LLM models.

    Each model must implement:
    - format_prompt
    - generate
    - parse_response
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a prompt from a single dataset sample.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate raw text response from the model.
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> str:
        """
        Parse model output into a normalized answer string.
        """
        pass

    def run(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run model inference on a list of samples.

        Returns:
            List of prediction records with metadata preserved.
        """
        outputs = []

        for sample in samples:
            prompt = self.format_prompt(sample)
            raw_response = self.generate(prompt)
            prediction = self.parse_response(raw_response)

            outputs.append({
                "group_id": sample["group_id"],
                "qid": sample["qid"],
                "qtype": sample["qtype"],
                "question": sample["question"],
                "golden_answers": sample.get("golden_answers"),
                "prediction": prediction,
                "model": self.name,
            })

        return outputs
