import os
import time
from typing import Dict, Any

import openai

from src.models.base import BaseModel


class GPTModel(BaseModel):
    """
    OpenAI GPT-based model wrapper.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.1",
        temperature: float = 0.0,
        max_tokens: int = 64,
        sleep_time: float = 0.5,
    ):
        super().__init__(name=model_name)

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sleep_time = sleep_time

        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Construct prompt from dataset sample.
        """
        question = sample["question"]

        contexts = []
        for key in ["document_a", "document_b", "document_c"]:
            if key in sample:
                title = sample[key].get("title", "")
                content = sample[key].get("contents", "")
                contexts.append(f"[{title}]\n{content}")

        context_text = "\n\n".join(contexts)

        prompt = (
            "Answer the following question using only the given documents.\n"
            "Answer briefly and precisely.\n\n"
            f"{context_text}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        return prompt

    def generate(self, prompt: str) -> str:
        """
        Call OpenAI API and return raw text response.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            text = response["choices"][0]["message"]["content"].strip()
            time.sleep(self.sleep_time)
            return text

        except Exception as e:
            print(f"[OpenAI ERROR] {e}")
            return ""

    def parse_response(self, response: str) -> str:
        """
        Normalize model response.
        """
        if response is None:
            return ""

        # Remove trailing punctuation or explanations
        response = response.strip()
        response = response.split("\n")[0]

        return response
