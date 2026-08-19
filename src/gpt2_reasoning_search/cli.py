"""Command-line entry point."""

import json
from pathlib import Path

import typer

from .config import ModelConfig, TrainConfig
from .model import GPT2ReasoningModel
from .prepare import prepare_token_corpora
from .tokenizer import load_tokenizer, train_tokenizer
from .train import train

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
