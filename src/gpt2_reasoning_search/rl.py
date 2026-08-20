"""Online GRPO-style reinforcement learning for bounded local-search QA."""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .agent import SearchAgent
from .checkpoint import load_checkpoint, load_model_weights, save_checkpoint
from .config import ModelConfig
from .evaluation import exact_match, strip_citation_markers, token_f1
from .judge import JudgeScore, QwenRewardJudge
from .model import GPT2ReasoningModel
from .retrieval import LocalWikipediaSearchProvider
from .schemas import AnswerRequest, AnswerResponse
from .train import optimizer_parameter_groups


@dataclass(frozen=True, slots=True)
class RewardWeights:
    answer_exact: float = 1.0
    answer_f1: float = 0.25
    citation_precision: float = 0.20
    citation_recall: float = 0.20
    citation_validity: float = 0.20
    valid_tool_calls: float = 0.10
    query_recovery: float = 0.10
    unnecessary_search: float = -0.10
    invalid_tool_call: float = -0.25
    search_cost: float = -0.02
    judge_answer_correctness: float = 0.20
    judge_evidence_support: float = 0.15
    judge_search_quality: float = 0.05
    judge_valid: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchRLConfig:
    checkpoint_directory: Path
    tokenizer_path: Path
    prompts_path: Path
    index_directory: Path
    output_directory: Path
    epochs: int = 1
    group_size: int = 4
    max_searches: int = 3
    learning_rate: float = 1e-6
    warmup_fraction: float = 0.03
    kl_coefficient: float = 0.02
    temperature: float = 0.8
    top_k: int = 50
    max_new_tokens: int = 512
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    save_every_steps: int = 100
    seed: int = 42
    enable_reranker: bool = False
    retrieval_device: str = "cpu"
    judge_model: str | None = None
    judge_revision: str | None = None
    judge_device: str = "cuda"
    time_budget_hours: float | None = None
    judge_max_input_tokens: int = 4096
    judge_max_new_tokens: int = 128
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.group_size < 2:
            raise ValueError("RL requires at least one epoch and two rollouts per group")
        if not 0 <= self.max_searches <= 3:
            raise ValueError("max_searches must be between zero and three")
        if self.learning_rate <= 0 or self.kl_coefficient < 0 or self.grad_clip <= 0:
            raise ValueError("invalid optimizer or KL configuration")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.temperature <= 0 or self.top_k < 0 or self.max_new_tokens < 1:
            raise ValueError("invalid sampling configuration")
        if self.save_every_steps < 1:
            raise ValueError("save_every_steps must be positive")
        if self.retrieval_device not in {"cpu", "cuda"}:
            raise ValueError("retrieval_device must be cpu or cuda")
        if self.judge_device not in {"cpu", "cuda"}:
            raise ValueError("judge_device must be cpu or cuda")
        if self.time_budget_hours is not None and (
            not math.isfinite(self.time_budget_hours) or self.time_budget_hours <= 0
        ):
            raise ValueError("time_budget_hours must be positive")
        if self.judge_max_input_tokens < 256 or self.judge_max_new_tokens < 16:
            raise ValueError("invalid judge token limits")
        if self.judge_model and not self.judge_revision:
            raise ValueError("judge_revision is required when judge_model is enabled")
        if self.judge_revision and re.fullmatch(r"[0-9a-f]{40}", self.judge_revision) is None:
            raise ValueError("judge_revision must be a full lowercase commit hash")


@dataclass(frozen=True, slots=True)
class RolloutSegment:
    token_ids: tuple[int, ...]
    prompt_length: int

    @property
    def action_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_length


@dataclass(frozen=True, slots=True)
class RewardResult:
    total: float
    components: dict[str, float]


class PolicyGenerator:
    """Record model-generated segments while satisfying SearchAgent's generation contract."""

    def __init__(
        self,
        model: GPT2ReasoningModel,
        tokenizer: Tokenizer,
        device: torch.device,
        *,
        temperature: float,
        top_k: int,
        maximum_new_tokens: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.maximum_new_tokens = maximum_new_tokens
        self.segments: list[RolloutSegment] = []
        self.outputs: list[str] = []

    def reset(self) -> None:
        self.segments.clear()
        self.outputs.clear()

    def _prompt_ids(self, prompt: str, completion_budget: int) -> list[int]:
        ids = self.tokenizer.encode(prompt).ids
        maximum = max(1, self.model.config.max_seq_len - completion_budget)
        if len(ids) <= maximum:
            return ids
        head = min(128, maximum // 4)
        return ids[:head] + ids[-(maximum - head) :]

    def __call__(self, prompt: str, max_new_tokens: int) -> tuple[str, int, int]:
        completion_budget = min(
            max_new_tokens,
            self.maximum_new_tokens,
            self.model.config.max_seq_len - 1,
        )
        prompt_ids = self._prompt_ids(prompt, completion_budget)
        if not prompt_ids:
            raise ValueError("RL prompt must contain at least one token")
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        eos = self.tokenizer.token_to_id("<|eos|>")
        end_tool = self.tokenizer.token_to_id("<|end_tool_call|>")
        stop_ids = {end_tool} if end_tool is not None else None
        with torch.inference_mode():
            output = self.model.generate(
                inputs,
                max_new_tokens=completion_budget,
                temperature=self.temperature,
                top_k=self.top_k,
                eos_token_id=eos,
                stop_token_ids=stop_ids,
            )
        token_ids = output[0].tolist()
        generated_ids = token_ids[len(prompt_ids) :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        self.segments.append(RolloutSegment(tuple(token_ids), len(prompt_ids)))
        self.outputs.append(text)
        return text, len(prompt_ids), len(generated_ids)


def stream_rl_prompts(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"RL prompt line {line_number} is missing question")
            if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                raise ValueError(f"RL prompt line {line_number} is missing answer")
            supporting = row.get("supporting_ids", [])
            if not isinstance(supporting, list) or not all(
                isinstance(value, str) for value in supporting
            ):
                raise ValueError(f"RL prompt line {line_number} has invalid supporting_ids")
            if "search_required" in row and not isinstance(row["search_required"], bool):
                raise ValueError(f"RL prompt line {line_number} has invalid search_required")
            yield row


def score_search_reward(
    row: dict[str, Any],
    response: AnswerResponse,
    raw_final_output: str,
    weights: RewardWeights | None = None,
    judge_score: JudgeScore | None = None,
) -> RewardResult:
    weights = weights or RewardWeights()
    claimed_ids = list(dict.fromkeys(re.findall(r"<\|citation\|>([^\s<]+)", raw_final_output)))
    returned_ids = {
        result.id for trace_entry in response.tool_trace for result in trace_entry.results
    }
    supporting_ids = set(map(str, row.get("supporting_ids", [])))
    grounded_claims = set(claimed_ids) & returned_ids
    answer = strip_citation_markers(response.answer, claimed_ids)
    answer_exact = exact_match(answer, str(row["answer"]))
    answer_f1 = token_f1(answer, str(row["answer"]))
    if supporting_ids:
        citation_precision = (
            len(grounded_claims & supporting_ids) / len(claimed_ids) if claimed_ids else 0.0
        )
        citation_recall = len(grounded_claims & supporting_ids) / len(supporting_ids)
    else:
        citation_precision = citation_recall = 0.0
    citation_validity = (
        len(set(claimed_ids) & returned_ids) / len(claimed_ids) if claimed_ids else 1.0
    )
    attempted = len(response.tool_trace)
    valid = sum(entry.status in {"ok", "empty", "error"} for entry in response.tool_trace)
    valid_rate = valid / attempted if attempted else 1.0
    invalid = sum(entry.status in {"invalid", "rejected"} for entry in response.tool_trace)
    search_required = bool(row.get("search_required", True))
    searched = response.searches_used > 0
    unnecessary = float(searched and not search_required)
    valid_entries = [
        entry
        for entry in response.tool_trace
        if entry.call is not None and entry.status in {"ok", "empty", "error"}
    ]
    recovered = 0.0
    if len(valid_entries) > 1 and answer_exact == 1.0:
        first_ids = {result.id for result in valid_entries[0].results}
        first_failed = valid_entries[0].status != "ok" or (
            bool(supporting_ids) and not first_ids & supporting_ids
        )
        later_succeeded = any(
            entry.status == "ok"
            and bool(entry.results)
            and (
                not supporting_ids or bool({result.id for result in entry.results} & supporting_ids)
            )
            for entry in valid_entries[1:]
        )
        distinct_queries = (
            len({entry.call.arguments.query.casefold() for entry in valid_entries if entry.call})
            > 1
        )
        recovered = float(first_failed and later_succeeded and distinct_queries)
    components = {
        "answer_exact": answer_exact,
        "answer_f1": answer_f1,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_validity": citation_validity,
        "valid_tool_calls": valid_rate,
        "query_recovery": recovered,
        "unnecessary_search": unnecessary,
        "invalid_tool_call": float(invalid),
        "search_cost": float(response.searches_used),
        "judge_answer_correctness": (
            judge_score.answer_correctness if judge_score and judge_score.valid else 0.0
        ),
        "judge_evidence_support": (
            judge_score.evidence_support if judge_score and judge_score.valid else 0.0
        ),
        "judge_search_quality": (
            judge_score.search_quality if judge_score and judge_score.valid else 0.0
        ),
        "judge_valid": float(bool(judge_score and judge_score.valid)),
    }
    total = sum(getattr(weights, name) * value for name, value in components.items())
    return RewardResult(total, components)


def group_advantages(rewards: Sequence[float], epsilon: float = 1e-4) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("group-relative advantages require at least two rewards")
    values = np.asarray(rewards, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("rewards must be finite")
    return ((values - values.mean()) / (values.std() + epsilon)).tolist()


def segment_policy_loss(
    policy: GPT2ReasoningModel,
    reference: GPT2ReasoningModel,
    segment: RolloutSegment,
    advantage: float,
    kl_coefficient: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if segment.action_tokens < 1:
        raise ValueError("rollout segment contains no model action tokens")
    tokens = torch.tensor([segment.token_ids], dtype=torch.long, device=device)
    policy_logits = policy(tokens).logits[:, :-1].float()
    targets = tokens[:, 1:]
    policy_log_probs = (
        F.log_softmax(policy_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    )
    with torch.no_grad():
        reference_logits = reference(tokens).logits[:, :-1].float()
        reference_log_probs = (
            F.log_softmax(reference_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        )
    start = max(0, segment.prompt_length - 1)
    action_log_probs = policy_log_probs[:, start:]
    reference_action_log_probs = reference_log_probs[:, start:]
    log_ratio = reference_action_log_probs - action_log_probs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    loss = -(advantage * action_log_probs - kl_coefficient * kl).mean()
    return loss, kl.detach().mean()


def _learning_rate_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_search_rl(config: SearchRLConfig, reward_weights: RewardWeights | None = None) -> Path:
    reward_weights = reward_weights or RewardWeights()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for search reinforcement learning")
    deadline = (
        time.perf_counter() + config.time_budget_hours * 3600
        if config.time_budget_hours is not None
        else None
    )
    prompts = list(stream_rl_prompts(config.prompts_path))
    if not prompts:
        raise ValueError("RL prompt dataset is empty")
    state = json.loads((config.checkpoint_directory / "state.json").read_text())
    base_config = ModelConfig(**state["config"]["model"])
    policy_config = ModelConfig(**{**base_config.to_dict(), "gradient_checkpointing": True})
    reference_config = ModelConfig(**{**base_config.to_dict(), "gradient_checkpointing": False})
    device = torch.device("cuda")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")

    policy = GPT2ReasoningModel(policy_config).to(device=device, dtype=torch.bfloat16)
    reference = GPT2ReasoningModel(reference_config).to(device=device, dtype=torch.bfloat16)
    load_model_weights(config.checkpoint_directory, policy, device)
    load_model_weights(config.checkpoint_directory, reference, device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(policy, config.weight_decay),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )
    total_steps = len(prompts) * config.epochs
    warmup_steps = round(total_steps * config.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_factor(step, total_steps, warmup_steps),
    )
    step = start_epoch = prompt_cursor = rollouts_seen = action_tokens_seen = 0
    if config.resume_from is not None:
        restored = load_checkpoint(config.resume_from, policy, optimizer, scheduler, device)
        step = int(restored["step"])
        progress = restored["mixture_state"]
        start_epoch = int(progress.get("epoch", 0))
        prompt_cursor = int(progress.get("prompt_cursor", 0))
        rollouts_seen = int(progress.get("rollouts", 0))
        action_tokens_seen = int(progress.get("total_action_tokens", 0))

    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
    generator = PolicyGenerator(
        policy,
        tokenizer,
        device,
        temperature=config.temperature,
        top_k=config.top_k,
        maximum_new_tokens=config.max_new_tokens,
    )
    provider = LocalWikipediaSearchProvider(
        config.index_directory,
        enable_reranker=config.enable_reranker,
        model_device=config.retrieval_device,
    )
    agent = SearchAgent(generator, provider)
    judge: QwenRewardJudge | None = None
    config.output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_directory / "metrics.jsonl"
    checkpoint_config = {
        **state["config"],
        "model": policy_config.to_dict(),
        "search_rl": {**asdict(config), "reward_weights": asdict(reward_weights)},
    }

    timed_out = False
    current_epoch = start_epoch
    try:
        if config.judge_model and config.judge_revision:
            judge = QwenRewardJudge(
                config.judge_model,
                config.judge_revision,
                device=config.judge_device,
                max_input_tokens=config.judge_max_input_tokens,
                max_new_tokens=config.judge_max_new_tokens,
            )
        with metrics_path.open("a") as metrics:
            for epoch in range(start_epoch, config.epochs):
                current_epoch = epoch
                for prompt_index, row in enumerate(prompts):
                    if epoch == start_epoch and prompt_index < prompt_cursor:
                        continue
                    if deadline is not None and time.perf_counter() >= deadline:
                        timed_out = True
                        break
                    rollout_records = []
                    for _ in range(config.group_size):
                        generator.reset()
                        response = asyncio.run(
                            agent.answer(
                                AnswerRequest(
                                    query=row["question"],
                                    search_mode="local",
                                    max_searches=config.max_searches,
                                )
                            )
                        )
                        raw_final = generator.outputs[-1] if generator.outputs else ""
                        judge_started = time.perf_counter()
                        judge_score = (
                            judge.score(row["question"], row["answer"], response)
                            if judge is not None
                            else None
                        )
                        judge_latency = (
                            time.perf_counter() - judge_started if judge is not None else 0.0
                        )
                        reward = score_search_reward(
                            row,
                            response,
                            raw_final,
                            reward_weights,
                            judge_score,
                        )
                        rollout_records.append(
                            (tuple(generator.segments), response, reward, judge_latency)
                        )
                    rewards = [record[2].total for record in rollout_records]
                    advantages = group_advantages(rewards)
                    segment_count = sum(len(record[0]) for record in rollout_records)
                    if segment_count == 0:
                        raise RuntimeError("RL rollout produced no trainable model segments")
                    policy.train()
                    optimizer.zero_grad(set_to_none=True)
                    mean_kl = 0.0
                    for advantage, (segments, _response, _reward, _judge_latency) in zip(
                        advantages, rollout_records, strict=True
                    ):
                        for segment in segments:
                            loss, kl = segment_policy_loss(
                                policy,
                                reference,
                                segment,
                                advantage,
                                config.kl_coefficient,
                                device,
                            )
                            (loss / segment_count).backward()
                            mean_kl += float(kl) / segment_count
                            action_tokens_seen += segment.action_tokens
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), config.grad_clip
                    )
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    rollouts_seen += config.group_size
                    prompt_cursor = prompt_index + 1
                    component_means = {
                        name: sum(record[2].components[name] for record in rollout_records)
                        / config.group_size
                        for name in rollout_records[0][2].components
                    }
                    record = {
                        "step": step,
                        "epoch": epoch,
                        "prompt_cursor": prompt_cursor,
                        "reward_mean": float(np.mean(rewards)),
                        "reward_std": float(np.std(rewards)),
                        "kl_mean": mean_kl,
                        "gradient_norm": float(grad_norm),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "rollouts": rollouts_seen,
                        "action_tokens": action_tokens_seen,
                        "judge_latency_seconds": float(
                            np.mean([rollout[3] for rollout in rollout_records])
                        ),
                        **{f"reward/{name}": value for name, value in component_means.items()},
                    }
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    if step % config.save_every_steps == 0:
                        save_checkpoint(
                            config.output_directory / f"step-{step:08d}",
                            policy,
                            optimizer,
                            scheduler,
                            step,
                            action_tokens_seen,
                            {
                                "epoch": epoch,
                                "prompt_cursor": prompt_cursor,
                                "rollouts": rollouts_seen,
                                "total_action_tokens": action_tokens_seen,
                            },
                            checkpoint_config,
                        )
                    has_remaining_work = prompt_cursor < len(prompts) or epoch + 1 < config.epochs
                    if (
                        has_remaining_work
                        and deadline is not None
                        and time.perf_counter() >= deadline
                    ):
                        timed_out = True
                        break
                if timed_out:
                    break
                prompt_cursor = 0
    finally:
        asyncio.run(agent.aclose())
        if judge is not None:
            judge.close()

    if timed_out:
        checkpoint = config.output_directory / f"step-{step:08d}"
        save_checkpoint(
            checkpoint,
            policy,
            optimizer,
            scheduler,
            step,
            action_tokens_seen,
            {
                "epoch": current_epoch,
                "prompt_cursor": prompt_cursor,
                "rollouts": rollouts_seen,
                "total_action_tokens": action_tokens_seen,
            },
            checkpoint_config,
        )
        return checkpoint

    final = config.output_directory / "final"
    save_checkpoint(
        final,
        policy,
        optimizer,
        scheduler,
        step,
        action_tokens_seen,
        {
            "epoch": config.epochs,
            "prompt_cursor": 0,
            "rollouts": rollouts_seen,
            "total_action_tokens": action_tokens_seen,
        },
        checkpoint_config,
    )
    return final
