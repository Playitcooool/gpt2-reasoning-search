"""Command-line entry point."""

import asyncio
import json
import os
from pathlib import Path

import typer

from .agent import SearchAgent
from .api import create_app
from .bootstrap import bootstrap_local_inputs
from .config import ModelConfig, TrainConfig
from .evaluation import (
    compare_proxy_runs,
    run_grounded_benchmark,
    score_grounded_records,
    score_language_model_records,
    score_reasoning_records,
    write_report,
)
from .evaluation import (
    stream_jsonl as stream_evaluation_jsonl,
)
from .experiment import write_experiment_plan
from .judge import DEFAULT_QWEN_JUDGE_MODEL, DEFAULT_QWEN_JUDGE_REVISION
from .model import GPT2ReasoningModel
from .prepare import load_evaluation_prompts, prepare_token_corpora
from .retrieval import (
    LocalWikipediaSearchProvider,
    SentenceTransformerEncoder,
    build_wikipedia_index,
)
from .rl import SearchRLConfig, train_search_rl
from .runner import ModelRunner
from .schemas import AnswerRequest
from .search import stream_jsonl_documents
from .sft import fine_tune_tools
from .smoke import tiny_overfit
from .tokenizer import load_tokenizer, train_tokenizer
from .train import train
from .trajectories import generate_trajectories, stream_jsonl
from .web_search import BraveWebSearchProvider, SQLiteSearchCache

app = typer.Typer(
    name="gpt2-reasoning-search",
    help="Train and serve a GPT-2-style reasoning model with search.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo("0.3.0")


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


@app.command("bootstrap-data")
def bootstrap_data_command(
    data_root: Path = typer.Option(Path("data")),
    manifest_output: Path = typer.Option(Path("artifacts/auto-data-manifest.json")),
    wikipedia_articles: int = typer.Option(5_000, min=1),
) -> None:
    """Download pinned public sources and create the fixed local training inputs."""
    report = bootstrap_local_inputs(data_root, manifest_output, wikipedia_articles)
    typer.echo(json.dumps(report, indent=2))


@app.command("prepare-data")
def prepare_data_command(
    tokenizer_path: Path = typer.Option(..., exists=True),
    manifest: Path = typer.Option(Path("config/datasets.json"), exists=True),
    output: Path = typer.Option(Path("data/processed")),
    reasoning_token_cap: int = typer.Option(2_000_000_000),
    general_token_cap: int = typer.Option(1_000_000_000),
    evaluation_prompts: Path | None = typer.Option(
        None, exists=True, help="JSONL or text prompts to exclude from training"
    ),
) -> None:
    """Stream, normalize, deduplicate, and tokenize the pinned corpora."""
    reports = prepare_token_corpora(
        load_tokenizer(tokenizer_path),
        manifest,
        output,
        reasoning_token_cap,
        general_token_cap,
        load_evaluation_prompts(evaluation_prompts),
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
    time_budget_hours: float = typer.Option(7.5, min=0.1),
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
    lexical_only: bool = typer.Option(False, help="Skip semantic embeddings and build BM25 only"),
    retrieval_config: Path = typer.Option(Path("config/retrieval.json"), exists=True),
    embedding_device: str | None = typer.Option(None, help="cpu, cuda, or auto when omitted"),
) -> None:
    """Build local BM25 and semantic indexes from Wikipedia-style JSONL."""
    settings = json.loads(retrieval_config.read_text())
    model = settings["embedding_model"]
    encoder = (
        None
        if lexical_only
        else SentenceTransformerEncoder(model["name"], model["revision"], embedding_device)
    )
    count = build_wikipedia_index(
        stream_jsonl_documents(documents),
        output,
        dense_encoder=encoder,
        retrieval_config=retrieval_config,
    )
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
    time_budget_hours: float = typer.Option(7.5, min=0.1),
    micro_batch_size: int = typer.Option(4, min=1),
    gradient_accumulation_steps: int = typer.Option(8, min=1),
    resume_from: Path | None = typer.Option(None, exists=True),
) -> None:
    """Fine-tune a pretrained checkpoint on search trajectories."""
    typer.echo(
        str(
            fine_tune_tools(
                checkpoint,
                tokenizer_path,
                trajectories,
                output,
                epochs=epochs,
                time_budget_hours=time_budget_hours,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                resume_from=resume_from,
            )
        )
    )


@app.command("rl-search")
def rl_search_command(
    checkpoint: Path = typer.Option(..., exists=True),
    tokenizer_path: Path = typer.Option(..., exists=True),
    prompts: Path = typer.Option(..., exists=True),
    index: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("checkpoints/search-rl")),
    epochs: int = typer.Option(1, min=1),
    time_budget_hours: float = typer.Option(7.5, min=0.1),
    group_size: int = typer.Option(4, min=2),
    max_searches: int = typer.Option(3, min=0, max=3),
    learning_rate: float = typer.Option(1e-6, min=1e-9),
    kl_coefficient: float = typer.Option(0.02, min=0.0),
    resume_from: Path | None = typer.Option(None, exists=True),
    enable_reranker: bool = typer.Option(False),
    retrieval_device: str = typer.Option("cpu"),
    llm_judge: bool = typer.Option(True, "--llm-judge/--no-llm-judge"),
    judge_model: str = typer.Option(DEFAULT_QWEN_JUDGE_MODEL),
    judge_revision: str | None = typer.Option(
        None, help="Pinned model revision; defaults only for Qwen3.5-2B"
    ),
    judge_device: str = typer.Option("cuda"),
) -> None:
    """Optimize QA and search behavior with grouped online RL rollouts."""
    resolved_judge_revision = judge_revision
    if llm_judge and resolved_judge_revision is None:
        if judge_model != DEFAULT_QWEN_JUDGE_MODEL:
            raise typer.BadParameter("custom --judge-model requires --judge-revision")
        resolved_judge_revision = DEFAULT_QWEN_JUDGE_REVISION
    config = SearchRLConfig(
        checkpoint_directory=checkpoint,
        tokenizer_path=tokenizer_path,
        prompts_path=prompts,
        index_directory=index,
        output_directory=output,
        epochs=epochs,
        time_budget_hours=time_budget_hours,
        group_size=group_size,
        max_searches=max_searches,
        learning_rate=learning_rate,
        kl_coefficient=kl_coefficient,
        resume_from=resume_from,
        enable_reranker=enable_reranker,
        retrieval_device=retrieval_device,
        judge_model=judge_model if llm_judge else None,
        judge_revision=resolved_judge_revision if llm_judge else None,
        judge_device=judge_device,
    )
    typer.echo(str(train_search_rl(config)))


def _load_agent(
    checkpoint: Path, tokenizer_path: Path, index: Path, enable_web: bool = True
) -> SearchAgent:
    runner = ModelRunner(checkpoint, tokenizer_path)
    local = LocalWikipediaSearchProvider(index, enable_reranker=True)
    key = os.getenv("BRAVE_SEARCH_API_KEY")
    web = (
        BraveWebSearchProvider(
            key,
            cache=SQLiteSearchCache(
                Path(os.getenv("GRS_SEARCH_CACHE", "artifacts/search-cache.sqlite3"))
            ),
        )
        if key and enable_web
        else None
    )
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


@app.command("experiment-plan")
def experiment_plan_command(
    output: Path = typer.Option(Path("artifacts/one-h100-plan.json")),
) -> None:
    """Write the fixed proxy-and-main schedule for one H100 day."""
    write_experiment_plan(output)
    typer.echo(str(output))


@app.command("smoke-overfit")
def smoke_overfit_command(
    steps: int = typer.Option(40, min=1), device: str = typer.Option("cpu")
) -> None:
    """Verify that a tiny model can overfit reasoning and tool-token patterns."""
    result = tiny_overfit(steps=steps, device=device)
    typer.echo(json.dumps(result, indent=2))
    if not result["passed"]:
        raise typer.Exit(1)


@app.command("score-reasoning")
def score_reasoning_command(
    predictions: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("artifacts/reasoning-report.json")),
) -> None:
    """Score JSONL predictions for GSM8K, MATH, code, and logic tasks."""
    report = score_reasoning_records(stream_evaluation_jsonl(predictions))
    write_report(report, output)
    typer.echo(json.dumps(report, indent=2))


@app.command("score-lm")
def score_lm_command(
    losses: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("artifacts/language-model-report.json")),
) -> None:
    """Aggregate token-weighted held-out losses and perplexity."""
    report = score_language_model_records(stream_evaluation_jsonl(losses))
    write_report(report, output)
    typer.echo(json.dumps(report, indent=2))


@app.command("score-grounded")
def score_grounded_command(
    predictions: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("artifacts/grounded-report.json")),
) -> None:
    """Score answer, retrieval, citation, and tool-use behavior."""
    report = score_grounded_records(stream_evaluation_jsonl(predictions))
    write_report(report, output)
    typer.echo(json.dumps(report, indent=2))


@app.command("benchmark-grounded")
def benchmark_grounded_command(
    examples: Path = typer.Argument(..., exists=True, readable=True),
    checkpoint: Path = typer.Option(..., exists=True),
    tokenizer_path: Path = typer.Option(..., exists=True),
    index: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("artifacts/grounded-predictions.jsonl")),
    modes: list[str] = typer.Option(["off", "local"], "--mode"),
    max_searches: int = typer.Option(3, min=0, max=3),
) -> None:
    """Run matched search-off/search-on grounded evaluation records."""
    allowed = {"off", "local", "web", "auto"}
    if not modes or any(mode not in allowed for mode in modes):
        raise typer.BadParameter("mode must be one of off, local, web, or auto")
    agent = _load_agent(
        checkpoint,
        tokenizer_path,
        index,
        enable_web=bool({"web", "auto"} & set(modes)),
    )

    async def run() -> int:
        try:
            return await run_grounded_benchmark(
                agent,
                stream_evaluation_jsonl(examples),
                output,
                modes,
                max_searches,
            )
        finally:
            await agent.aclose()

    count = asyncio.run(run())
    typer.echo(json.dumps({"records": count, "output": str(output)}))


@app.command("compare-proxies")
def compare_proxies_command(
    runs: list[Path] = typer.Argument(..., exists=True),
    output: Path = typer.Option(Path("artifacts/proxy-comparison.json")),
) -> None:
    """Compare the matched 0%, 30%, and 70% proxy training runs."""
    report = compare_proxy_runs(runs)
    write_report(report, output)
    typer.echo(json.dumps(report, indent=2))
