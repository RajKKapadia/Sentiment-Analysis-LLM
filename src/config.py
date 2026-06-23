"""Configuration for decoder-only IMDb sentiment classifier.

Edit this file first when you want to change model size, sequence length,
split percentages, batch size, or training hyperparameters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    dataset_name: str = "stanfordnlp/imdb"
    tokenizer_name: str = "gpt2"

    # We combine IMDb's labeled train + test splits, shuffle, then create these splits.
    train_pct: float = 0.80
    valid_pct: float = 0.10
    test_pct: float = 0.10

    block_size: int = 512
    seed: int = 1337

    # Set to a small integer like 2000 for a quick smoke test. None uses all 50k labeled reviews.
    max_samples: Optional[int] = None

    data_dir: str = "data"
    processed_dir: str = "data/processed/imdb_gpt2"

    # GPT-2/tiktoken end-of-text token. We reuse it as PAD because GPT-2 BPE has no native pad token.
    eos_token_id: int = 50256
    pad_token_id: int = 50256


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.15
    num_classes: int = 2
    bias: bool = True


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "auto"  # auto, cuda, cpu

    epochs: int = 8
    batch_size: int = 32
    num_workers: int = 2
    gradient_accumulation_steps: int = 1

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-4
    weight_decay: float = 0.10
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200

    eval_interval_steps: int = 250
    log_interval_steps: int = 25

    # Use "bf16" on RTX 30/40/50 class GPUs if supported, otherwise use "fp16" or "none".
    mixed_precision: str = "bf16"  # bf16, fp16, none
    compile_model: bool = False

    run_name: str = "imdb_decoder_classifier"
    # Base directory. Each train.py run writes into a timestamped child folder.
    out_dir: str = "runs/imdb_decoder_classifier"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        # Keep these tied by default. If you change data.block_size, model gets the same context length.
        self.model.block_size = self.data.block_size
        self.model.vocab_size = 50257
        self.train.seed = self.data.seed
        self.validate()

    def validate(self) -> None:
        split_sum = self.data.train_pct + self.data.valid_pct + self.data.test_pct
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Split percentages must sum to 1.0, got {split_sum:.6f}. "
                "Edit train_pct/valid_pct/test_pct in config.py."
            )
        if self.model.n_embd % self.model.n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        if self.data.block_size <= 0:
            raise ValueError("data.block_size must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


CFG = AppConfig()
