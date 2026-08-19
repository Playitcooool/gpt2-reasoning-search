"""Command-line entry point."""

import asyncio
import json
import os
from pathlib import Path

import typer

from .agent import SearchAgent
from .api import create_app
from .config import ModelConfig, TrainConfig
from .model import GPT2ReasoningModel
from .prepare import prepare_token_corpora
from .runner import ModelRunner
from .schemas import AnswerRequest
from .search import (
    BraveWebSearchProvider,
    LocalWikipediaSearchProvider,
    build_wikipedia_index,
    stream_jsonl_documents,
)
from .sft import fine_tune_tools
from .tokenizer import load_tokenizer, train_tokenizer
from .train import train
from .trajectories import generate_trajectories, stream_jsonl

app = typer.Typer(
    name="gpt2-reasoning-search",
    help="Train and serve a GPT-2-style reasoning model with search.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo("0.1.0")


@app.command("model-info")
def model_info(preset: str = typer.Option("main-350m", help="main-350m or proxy-124m")) -> None:
    """Print model configuration and exact parameter count."""
    config = ModelConfig.preset(preset)  # type: ignore[arg-type]
    model = GPT2ReasoningModel(config)
    typer.echo(
        json.dumps({"config": config.to_dict(), "parameters": model.parameter_count()}, indent=2)
    )


@app.command("train-tokenizer")
def tokenizer_command(
    inputs: list[Path] = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("artifacts/tokenizer.json")),
    vocab_size: int = typer.Option(50_304),
) -> None:
    """Train the project BPE tokenizer from text files."""
    train_tokenizer(inputs, output, vocab_size)
    typer.echo(str(output))


@app.command("prepare-data")
def prepare_data_command(
    tokenizer_path: Path = typer.Option(..., exists=True),
    manifest: Path = typer.Option(Path("config/datasets.json"), exists=True),
    output: Path = typer.Option(Path("data/processed")),
    reasoning_token_cap: int = typer.Option(2_000_000_000),
    general_token_cap: int = typer.Option(1_000_000_000),
) -> None:
    """Stream, normalize, deduplicate, and tokenize the pinned corpora."""
    reports = prepare_token_corpora(
        load_tokenizer(tokenizer_path),
        manifest,
        output,
        reasoning_token_cap,
        general_token_cap,
    )
    typer.echo(json.dumps(reports, indent=2))


@app.command("pretrain")
def pretrain_command(
    reasoning_tokens: Path = typer.Option(..., exists=True),
    general_tokens: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("checkpoints/main-350m")),
    preset: str = typer.Option("main-350m"),
    reasoning_ratio: float = typer.Option(0.70),
    max_tokens: int = typer.Option(2_500_000_000),
    time_budget_hours: float = typer.Option(18.0),
    resume_from: Path | None = typer.Option(None),
) -> None:
    """Run calibrated single-GPU pretraining."""
    config = TrainConfig(
        output_dir=output,
        reasoning_tokens=reasoning_tokens,
        general_tokens=general_tokens,
        model_preset=preset,  # type: ignore[arg-type]
        reasoning_ratio=reasoning_ratio,
        max_tokens=max_tokens,
        time_budget_hours=time_budget_hours,
    )
    typer.echo(str(train(config, resume_from)))


@app.command("build-index")
def build_index_command(
    documents: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("artifacts/wiki-index")),
) -> None:
    """Build a local BM25 index from Wikipedia-style JSONL documents."""
    count = build_wikipedia_index(stream_jsonl_documents(documents), output)
    typer.echo(json.dumps({"chunks": count, "index": str(output)}))


@app.command("make-trajectories")
def make_trajectories_command(
    examples: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("data/processed/tool-trajectories.jsonl")),
) -> None:
    """Create normalized, evidence-grounded tool-use trajectories."""
    count = generate_trajectories(stream_jsonl(examples), output)
    typer.echo(json.dumps({"examples": count, "output": str(output)}))


@app.command("sft-tools")
def sft_tools_command(
    checkpoint: Path = typer.Option(..., exists=True),
    tokenizer_path: Path = typer.Option(..., exists=True),
    trajectories: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("checkpoints/tool-sft")),
    epochs: int = typer.Option(1, min=1),
) -> None:
    """Fine-tune a pretrained checkpoint on search trajectories."""
    typer.echo(
        str(fine_tune_tools(checkpoint, tokenizer_path, trajectories, output, epochs=epochs))
    )


def _load_agent(
    checkpoint: Path, tokenizer_path: Path, index: Path, enable_web: bool = True
) -> SearchAgent:
    runner = ModelRunner(checkpoint, tokenizer_path)
    local = LocalWikipediaSearchProvider(index)
    key = os.getenv("BRAVE_SEARCH_API_KEY")
    web = BraveWebSearchProvider(key) if key and enable_web else None
    return SearchAgent(runner.generate, local, web)


@app.command("ask")
def ask_command(
    query: str = typer.Argument(...),
    checkpoint: Path = typer.Option(..., exists=True),
    tokenizer_path: Path = typer.Option(..., exists=True),
    index: Path = typer.Option(..., exists=True),
    search_mode: str = typer.Option("auto"),
    max_searches: int = typer.Option(3, min=0, max=3),
) -> None:
    """Answer one question using the bounded search controller."""
    agent = _load_agent(checkpoint, tokenizer_path, index)
    request = AnswerRequest(query=query, search_mode=search_mode, max_searches=max_searches)  # type: ignore[arg-type]
    response = asyncio.run(agent.answer(request))
    typer.echo(response.model_dump_json(indent=2))


@app.command("serve")
def serve_command(
    checkpoint: Path = typer.Option(..., exists=True),
    tokenizer_path: Path = typer.Option(..., exists=True),
    index: Path = typer.Option(..., exists=True),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Serve the health and grounded-answer HTTP endpoints."""
    import uvicorn

    uvicorn.run(create_app(_load_agent(checkpoint, tokenizer_path, index)), host=host, port=port)
