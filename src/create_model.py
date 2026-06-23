"""Decoder-only Transformer with a classification head for sentiment prediction."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CFG, ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
            self.register_buffer("bias", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head

        q = q.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderSentimentClassifier(nn.Module):
    """GPT-like decoder backbone + linear sentiment classification head.

    The model pools the hidden state at the last non-padding token. Because the
    decoder is causal, that token can attend to the full review prefix.
    """

    def __init__(self, config: ModelConfig = CFG.model) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.classifier = nn.Linear(config.n_embd, config.num_classes, bias=True)

        self.apply(self._init_weights)

        # GPT-style scaled init for residual projections.
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        device = input_ids.device
        batch_size, seq_len = input_ids.size()

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence length {seq_len}; block size is {self.config.block_size}"
            )

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if attention_mask is None:
            last_token_idx = torch.full(
                (batch_size,), seq_len - 1, dtype=torch.long, device=device
            )
        else:
            # Right padding is used. Last useful token = sum(mask)-1.
            last_token_idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1

        pooled = x[torch.arange(batch_size, device=device), last_token_idx]
        logits = self.classifier(pooled)

        output = {"logits": logits, "pooled": pooled}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        return output

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def crop_block_size(self, block_size: int) -> None:
        """Optional helper if you train with smaller context after creating a larger model."""
        if block_size > self.config.block_size:
            raise ValueError("Cannot increase block size with crop_block_size")
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]


def create_model(config: ModelConfig = CFG.model) -> DecoderSentimentClassifier:
    model = DecoderSentimentClassifier(config)
    print("Created DecoderSentimentClassifier")
    print(f"Config: {asdict(config)}")
    print(f"Parameters: {model.get_num_params() / 1e6:.2f}M")
    return model


if __name__ == "__main__":
    create_model(CFG.model)
