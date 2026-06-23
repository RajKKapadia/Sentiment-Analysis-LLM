"""Train the decoder-only IMDb sentiment classifier and save metrics/checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import CFG, AppConfig
from src.create_model import create_model

RUN_TIMESTAMP_FORMAT = "%H_%M_%d_%m_%Y"


class SentimentTensorDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.input_ids = payload["input_ids"]
        self.attention_mask = payload["attention_mask"]
        self.labels = payload["labels"]

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_amp_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "none":
        return nullcontext()
    if mixed_precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        print("bf16 requested but not supported; falling back to fp16 autocast.")
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    if mixed_precision == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError("mixed_precision must be one of: bf16, fp16, none")


def compute_binary_metrics(
    preds: torch.Tensor, labels: torch.Tensor, loss: float
) -> dict[str, float | int | list[list[int]]]:
    preds = preds.cpu().long()
    labels = labels.cpu().long()

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())

    total = max(1, int(labels.numel()))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, cfg: AppConfig
) -> dict:
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with get_amp_context(device, cfg.train.mixed_precision):
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

        logits = out["logits"]
        loss = out["loss"]
        preds = logits.argmax(dim=-1)

        losses.append(loss.detach().float().item() * labels.size(0))
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    total_examples = sum(x.numel() for x in all_labels)
    avg_loss = sum(losses) / max(1, total_examples)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metrics = compute_binary_metrics(preds, labels, avg_loss)
    metrics["num_examples"] = int(total_examples)
    model.train()
    return metrics


def get_lr(step: int, total_steps: int, cfg: AppConfig) -> float:
    if step < cfg.train.warmup_steps:
        return cfg.train.learning_rate * (step + 1) / max(1, cfg.train.warmup_steps)
    if step >= total_steps:
        return cfg.train.min_learning_rate

    decay_ratio = (step - cfg.train.warmup_steps) / max(
        1, total_steps - cfg.train.warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.train.min_learning_rate + coeff * (
        cfg.train.learning_rate - cfg.train.min_learning_rate
    )


def configure_optimizer(
    model: torch.nn.Module, cfg: AppConfig
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        optim_groups,
        lr=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
    )


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def create_timestamped_run_dir(base_dir: str | Path) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
    suffix = 1
    while True:
        dirname = timestamp if suffix == 1 else f"{timestamp}_{suffix}"
        run_dir = base_dir / dirname
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            suffix += 1


def resolve_default_checkpoint(
    base_dir: str | Path, checkpoint_name: str = "best.pt"
) -> Path:
    base_dir = Path(base_dir)
    if base_dir.exists():
        run_checkpoints = [
            path / checkpoint_name
            for path in base_dir.iterdir()
            if path.is_dir() and (path / checkpoint_name).exists()
        ]
        if run_checkpoints:
            return max(run_checkpoints, key=lambda path: path.stat().st_mtime)

    return base_dir / checkpoint_name


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: AppConfig,
    epoch: int,
    global_step: int,
    best_valid_f1: float,
    metrics: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(cfg.model),
        "data_config": asdict(cfg.data),
        "train_config": asdict(cfg.train),
        "epoch": epoch,
        "global_step": global_step,
        "best_valid_f1": best_valid_f1,
        "metrics": metrics or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train(
    cfg: AppConfig = CFG,
    resume_from: Optional[str] = None,
    description: str = "",
) -> Path:
    cfg = deepcopy(cfg)
    out_dir = create_timestamped_run_dir(cfg.train.out_dir)
    cfg.train.out_dir = str(out_dir)

    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    cfg.save_json(out_dir / "config_snapshot.json")

    processed_dir = Path(cfg.data.processed_dir)
    train_path = processed_dir / "train.pt"
    valid_path = processed_dir / "valid.pt"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_dir}. Run: python prepare_dataset.py"
        )

    train_ds = SentimentTensorDataset(train_path)
    valid_ds = SentimentTensorDataset(valid_path)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = create_model(cfg.model).to(device)
    if cfg.train.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = configure_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and cfg.train.mixed_precision == "fp16")
    )

    global_step = 0
    start_epoch = 0
    best_valid_f1 = -1.0

    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, optimizer, device)
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_valid_f1 = float(checkpoint.get("best_valid_f1", -1.0))
        print(f"Resumed from {resume_from} at epoch={start_epoch}, step={global_step}")

    total_optimizer_steps = (
        math.ceil(len(train_loader) / cfg.train.gradient_accumulation_steps)
        * cfg.train.epochs
    )
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "training_summary.json"

    print(f"Device: {device}")
    print(f"Run directory: {out_dir}")
    print(f"Train examples: {len(train_ds)}, Valid examples: {len(valid_ds)}")
    print(f"Total optimizer steps: {total_optimizer_steps}")

    model.train()
    running_loss = 0.0
    running_count = 0
    train_start_time = time.time()

    for epoch in range(start_epoch, cfg.train.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            lr = get_lr(global_step, total_optimizer_steps, cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with get_amp_context(device, cfg.train.mixed_precision):
                out = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = out["loss"] / cfg.train.gradient_accumulation_steps

            scaler.scale(loss).backward()

            should_step = (
                (batch_idx + 1) % cfg.train.gradient_accumulation_steps == 0
            ) or (batch_idx + 1 == len(train_loader))
            if should_step:
                if cfg.train.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.train.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                batch_loss = (
                    loss.detach().float().item() * cfg.train.gradient_accumulation_steps
                )
                running_loss += batch_loss
                running_count += 1

                if global_step % cfg.train.log_interval_steps == 0:
                    avg_train_loss = running_loss / max(1, running_count)
                    pbar.set_postfix(
                        {"loss": f"{avg_train_loss:.4f}", "lr": f"{lr:.2e}"}
                    )
                    append_jsonl(
                        metrics_path,
                        {
                            "type": "train",
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": avg_train_loss,
                            "lr": lr,
                            "time_sec": round(time.time() - train_start_time, 2),
                        },
                    )
                    running_loss = 0.0
                    running_count = 0

                if global_step % cfg.train.eval_interval_steps == 0:
                    valid_metrics = evaluate(model, valid_loader, device, cfg)
                    valid_record = {
                        "type": "valid",
                        "epoch": epoch + 1,
                        "step": global_step,
                        **valid_metrics,
                        "time_sec": round(time.time() - train_start_time, 2),
                    }
                    append_jsonl(metrics_path, valid_record)
                    print(
                        f"\nvalid step={global_step}: "
                        f"loss={valid_metrics['loss']:.4f}, "
                        f"acc={valid_metrics['accuracy']:.4f}, "
                        f"f1={valid_metrics['f1']:.4f}"
                    )

                    save_checkpoint(
                        out_dir / "latest.pt",
                        model,
                        optimizer,
                        cfg,
                        epoch=epoch,
                        global_step=global_step,
                        best_valid_f1=best_valid_f1,
                        metrics=valid_metrics,
                    )

                    if valid_metrics["f1"] > best_valid_f1:
                        best_valid_f1 = float(valid_metrics["f1"])
                        save_checkpoint(
                            out_dir / "best.pt",
                            model,
                            optimizer,
                            cfg,
                            epoch=epoch,
                            global_step=global_step,
                            best_valid_f1=best_valid_f1,
                            metrics=valid_metrics,
                        )
                        print(
                            f"Saved new best checkpoint with valid_f1={best_valid_f1:.4f}"
                        )

        # End-of-epoch validation, useful if eval_interval_steps is larger than one epoch.
        valid_metrics = evaluate(model, valid_loader, device, cfg)
        append_jsonl(
            metrics_path,
            {
                "type": "valid_epoch_end",
                "epoch": epoch + 1,
                "step": global_step,
                **valid_metrics,
                "time_sec": round(time.time() - train_start_time, 2),
            },
        )

        save_checkpoint(
            out_dir / "latest.pt",
            model,
            optimizer,
            cfg,
            epoch=epoch + 1,
            global_step=global_step,
            best_valid_f1=best_valid_f1,
            metrics=valid_metrics,
        )

        if valid_metrics["f1"] > best_valid_f1:
            best_valid_f1 = float(valid_metrics["f1"])
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                cfg,
                epoch=epoch + 1,
                global_step=global_step,
                best_valid_f1=best_valid_f1,
                metrics=valid_metrics,
            )

    final_summary = {
        "description": description,
        "best_valid_f1": best_valid_f1,
        "global_step": global_step,
        "epochs": cfg.train.epochs,
        "total_time_sec": round(time.time() - train_start_time, 2),
        "out_dir": str(out_dir),
        "best_checkpoint": str(out_dir / "best.pt"),
        "latest_checkpoint": str(out_dir / "latest.pt"),
    }
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))
    return out_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train IMDb decoder sentiment classifier."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Free-text run description saved only in training_summary.json.",
    )
    args = parser.parse_args()
    train(CFG, resume_from=args.resume_from, description=args.description)


if __name__ == "__main__":
    main()
