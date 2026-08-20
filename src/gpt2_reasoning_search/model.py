"""Modern GPT-2-style causal decoder with GQA and cached generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig

KeyValueCache = tuple[Tensor, Tensor]


@dataclass(slots=True)
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: tuple[KeyValueCache, ...] | None = None


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None, :, :], persistent=False)

    def apply_rotary(self, x: Tensor, position_offset: int = 0) -> Tensor:
        seq_len = x.shape[-2]
        end = position_offset + seq_len
        if end > self.cos.shape[-2]:
            raise ValueError("rotary position exceeds configured context length")
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        cos = self.cos[:, :, position_offset:end].to(dtype=x.dtype)
        sin = self.sin[:, :, position_offset:end].to(dtype=x.dtype)
        rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary = rotary
        projection_size = (config.n_heads + 2 * config.n_kv_heads) * self.head_dim
        self.qkv = nn.Linear(config.d_model, projection_size, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: Tensor,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KeyValueCache | None]:
        batch, seq_len, width = x.shape
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        q, k, v = self.qkv(x).split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        position_offset = past_key_value[0].shape[-2] if past_key_value is not None else 0
        q = self.rotary.apply_rotary(q, position_offset)
        k = self.rotary.apply_rotary(k, position_offset)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)
        attention_mask = None
        if past_key_value is not None and seq_len > 1:
            attention_mask = torch.ones(
                (seq_len, k.shape[-2]), dtype=torch.bool, device=x.device
            ).tril(diagonal=position_offset)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=past_key_value is None and seq_len > 1,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )
        output = self.out(y.transpose(1, 2).contiguous().view(batch, seq_len, width))
        return output, (k, v) if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(config.d_model, 2 * config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config, rotary)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: Tensor,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KeyValueCache | None]:
        attention, present = self.attn(self.attn_norm(x), past_key_value, use_cache)
        x = x + attention
        return x + self.mlp(self.mlp_norm(x)), present


class GPT2ReasoningModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, rotary) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=scale)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        past_key_values: tuple[KeyValueCache, ...] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        past_length = past_key_values[0][0].shape[-2] if past_key_values else 0
        if past_length + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured maximum")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one entry per layer")
        if self.training and use_cache:
            raise ValueError("KV caching is only supported during inference")
        x = self.token_embedding(input_ids)
        presents: list[KeyValueCache] = []
        for index, block in enumerate(self.blocks):
            past = past_key_values[index] if past_key_values is not None else None
            if self.config.gradient_checkpointing and self.training:
                def custom_forward(hidden: Tensor, current_block: Block = block) -> Tensor:
                    return current_block(hidden)[0]

                x = checkpoint(custom_forward, x, use_reentrant=False)
                present = None
            else:
                x, present = block(x, past, use_cache)
            if present is not None:
                presents.append(present)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(presents) if use_cache else None,
        )

    @staticmethod
    def _sample(logits: Tensor, temperature: float, top_k: int) -> Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k > 0:
            cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int | None = None,
        stop_token_ids: set[int] | None = None,
    ) -> Tensor:
        self.eval()
        output = input_ids[:, -self.config.max_seq_len :]
        past: tuple[KeyValueCache, ...] | None = None
        context = output
        stop_ids = set(stop_token_ids or ())
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)
        for _ in range(max_new_tokens):
            result = self(context, past_key_values=past, use_cache=True)
            next_token = self._sample(result.logits[:, -1], temperature, top_k)
            output = torch.cat((output, next_token), dim=1)
            if stop_ids and all(token.item() in stop_ids for token in next_token):
                break
            past = result.past_key_values
            if past is not None and past[0][0].shape[-2] >= self.config.max_seq_len:
                context = output[:, -self.config.max_seq_len :]
                past = None
            else:
                context = next_token
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
