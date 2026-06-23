"""Download IMDb, tokenize with tiktoken GPT-2 BPE, and save train/valid/test tensors.

Output files under CFG.data.processed_dir:
    train.pt, valid.pt, test.pt       -> tensor datasets
    train.jsonl, valid.jsonl, test.jsonl -> raw aligned text/label metadata
    meta.json                        -> dataset/tokenizer/config metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tiktoken
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from src.config import CFG, AppConfig

LABEL_NAMES = {0: "negative", 1: "positive"}


def _encode_and_pad(
    text: str, enc: Any, block_size: int, eos_token_id: int, pad_token_id: int
) -> tuple[list[int], list[int], int]:
    token_ids = enc.encode_ordinary(text)

    # Ensure the final useful position can represent the whole review.
    token_ids = token_ids[: max(0, block_size - 1)] + [eos_token_id]
    original_len = len(token_ids)

    if len(token_ids) < block_size:
        pad_len = block_size - len(token_ids)
        input_ids = token_ids + [pad_token_id] * pad_len
        attention_mask = [1] * len(token_ids) + [0] * pad_len
    else:
        input_ids = token_ids[:block_size]
        attention_mask = [1] * block_size
        original_len = block_size

    return input_ids, attention_mask, original_len


def _split_sizes(n: int, train_pct: float, valid_pct: float) -> tuple[int, int, int]:
    n_train = int(n * train_pct)
    n_valid = int(n * valid_pct)
    n_test = n - n_train - n_valid
    return n_train, n_valid, n_test


def _write_split(
    rows: list[dict[str, Any]],
    split_name: str,
    out_dir: Path,
    enc: Any,
    cfg: AppConfig,
) -> None:
    input_ids = []
    attention_masks = []
    labels = []

    raw_path = out_dir / f"{split_name}.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for item in tqdm(rows, desc=f"Tokenizing {split_name}"):
            text = str(item["text"])
            label = int(item["label"])
            ids, mask, token_len = _encode_and_pad(
                text=text,
                enc=enc,
                block_size=cfg.data.block_size,
                eos_token_id=cfg.data.eos_token_id,
                pad_token_id=cfg.data.pad_token_id,
            )
            input_ids.append(ids)
            attention_masks.append(mask)
            labels.append(label)

            raw_f.write(
                json.dumps(
                    {
                        "text": text,
                        "label": label,
                        "label_name": LABEL_NAMES[label],
                        "token_len": token_len,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    tensor_payload = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    torch.save(tensor_payload, out_dir / f"{split_name}.pt")


def prepare_dataset(cfg: AppConfig = CFG, force: bool = False) -> Path:
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = [
        out_dir / "train.pt",
        out_dir / "valid.pt",
        out_dir / "test.pt",
        out_dir / "meta.json",
    ]
    if all(p.exists() for p in expected) and not force:
        print(f"Prepared dataset already exists at {out_dir}. Use --force to rebuild.")
        return out_dir

    print(f"Loading dataset: {cfg.data.dataset_name}")
    ds = load_dataset(cfg.data.dataset_name)

    # IMDb from HF is usually train/test. We combine labeled splits so config percentages control all splits.
    combined = concatenate_datasets([ds["train"], ds["test"]])
    combined = combined.shuffle(seed=cfg.data.seed)

    if cfg.data.max_samples is not None:
        max_samples = min(cfg.data.max_samples, len(combined))
        combined = combined.select(range(max_samples))

    n_total = len(combined)
    n_train, n_valid, n_test = _split_sizes(
        n_total, cfg.data.train_pct, cfg.data.valid_pct
    )

    train_rows = [combined[i] for i in range(0, n_train)]
    valid_rows = [combined[i] for i in range(n_train, n_train + n_valid)]
    test_rows = [combined[i] for i in range(n_train + n_valid, n_total)]

    enc = tiktoken.get_encoding(cfg.data.tokenizer_name)

    print(
        f"Writing processed dataset to {out_dir}\n"
        f"Total={n_total}, train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}, "
        f"block_size={cfg.data.block_size}"
    )

    _write_split(train_rows, "train", out_dir, enc, cfg)
    _write_split(valid_rows, "valid", out_dir, enc, cfg)
    _write_split(test_rows, "test", out_dir, enc, cfg)

    meta = {
        "dataset_name": cfg.data.dataset_name,
        "tokenizer_name": cfg.data.tokenizer_name,
        "vocab_size": cfg.model.vocab_size,
        "block_size": cfg.data.block_size,
        "eos_token_id": cfg.data.eos_token_id,
        "pad_token_id": cfg.data.pad_token_id,
        "label_names": LABEL_NAMES,
        "num_classes": cfg.model.num_classes,
        "split_sizes": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
        "split_percentages": {
            "train_pct": cfg.data.train_pct,
            "valid_pct": cfg.data.valid_pct,
            "test_pct": cfg.data.test_pct,
        },
        "seed": cfg.data.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cfg.save_json(out_dir / "config_snapshot.json")

    print("Dataset preparation complete.")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare IMDb sentiment dataset for decoder classifier."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if processed files already exist.",
    )
    args = parser.parse_args()
    prepare_dataset(CFG, force=args.force)


if __name__ == "__main__":
    main()
