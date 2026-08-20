"""Checkpoint-backed text generation interface."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .checkpoint import load_model_weights
from .config import ModelConfig
from .model import GPT2ReasoningModel


class ModelRunner:
    def __init__(
        self, checkpoint_directory: Path, tokenizer_path: Path, device: str = "auto"
    ) -> None:
        state = json.loads((checkpoint_directory / "state.json").read_text())
        model_config = ModelConfig(**state["config"]["model"])
        resolved_device = (
            torch.device("cuda")
            if device == "auto" and torch.cuda.is_available()
            else torch.device("cpu" if device == "auto" else device)
        )
        self.device = resolved_device
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.model = GPT2ReasoningModel(model_config).to(resolved_device)
        load_model_weights(checkpoint_directory, self.model, resolved_device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512) -> tuple[str, int, int]:
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        eos = self.tokenizer.token_to_id("<|eos|>")
        output = self.model.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos)
        new_ids = output[0, input_ids.shape[1] :].tolist()
        return (
            self.tokenizer.decode(new_ids, skip_special_tokens=False),
            len(encoded.ids),
            len(new_ids),
        )
