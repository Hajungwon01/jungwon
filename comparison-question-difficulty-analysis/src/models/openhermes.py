import torch
from typing import Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.base import BaseModel


class OpenHermesModel(BaseModel):
    """
    OpenHermes (LLaMA-based) model wrapper using HuggingFace Transformers.
    """

    def __init__(
        self,
        model_name: str = "teknium/OpenHermes-2.5-Mistral-7B",
        device: str = "cuda",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ):
        super().__init__(name=model_name)

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model.eval()

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
        Run local LLM inference.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return decoded

    def parse_response(self, response: str) -> str:
        """
        Extract answer text from model output.
        """
        if response is None:
            return ""

        # 모델 출력에서 Answer 이후만 추출
        if "Answer:" in response:
            response = response.split("Answer:")[-1]

        response = response.strip()
        response = response.split("\n")[0]

        return response
