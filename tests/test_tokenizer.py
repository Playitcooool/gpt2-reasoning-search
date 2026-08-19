from pathlib import Path

from gpt2_reasoning_search.tokenizer import SPECIAL_TOKENS, load_tokenizer, train_tokenizer


def test_tokenizer_round_trip_and_boundary_tokens(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Reason carefully: 2 + 2 = 4.\n"
        "Unicode survives normalization: café, 中文, and emoji 🧠.\n"
        "A tool result can support a cited answer.\n"
    )
    output = tmp_path / "tokenizer.json"

    tokenizer = train_tokenizer([corpus], output, vocab_size=384, min_frequency=1)
    restored = load_tokenizer(output)
    text = "Reasoning with café, 中文, and 🧠."
    encoded = restored.encode(text)

    assert output.is_file()
    assert restored.decode(encoded.ids) == text
    assert encoded.ids[0] == restored.token_to_id("<|bos|>")
    assert encoded.ids[-1] == restored.token_to_id("<|eos|>")
    assert tokenizer.get_vocab_size() == restored.get_vocab_size()


def test_all_protocol_tokens_are_atomic_special_tokens(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ordinary training text repeated ordinary training text")
    tokenizer = train_tokenizer([corpus], tmp_path / "tokenizer.json", vocab_size=320)

    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        encoding = tokenizer.encode(token, add_special_tokens=False)
        assert token_id is not None
        assert encoding.ids == [token_id]
