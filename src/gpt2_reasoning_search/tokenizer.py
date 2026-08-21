"""Tokenizer training and loading."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|problem|>",
    "<|reasoning|>",
    "<|answer|>",
    "<|tool_call|>",
    "<|end_tool_call|>",
    "<|tool_result|>",
    "<|end_tool_result|>",
    "<|citation|>",
]


def train_tokenizer(
    files: list[Path], output: Path, vocab_size: int = 50_304, min_frequency: int = 2
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(path) for path in files], trainer)
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    if bos is None or eos is None:
        raise RuntimeError("required boundary tokens were not created")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>", special_tokens=[("<|bos|>", bos), ("<|eos|>", eos)]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".partial",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        tokenizer.save(str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return tokenizer


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode_documents(tokenizer: Tokenizer, texts: Iterable[str]) -> Iterable[list[int]]:
    for text in texts:
        if text.strip():
            yield tokenizer.encode(text).ids
