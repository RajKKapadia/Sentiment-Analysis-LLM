"""Evaluate a saved checkpoint on the test split and save metrics/predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CFG, AppConfig, ModelConfig
from src.create_model import DecoderSentimentClassifier
from src.train import (
    SentimentTensorDataset,
    compute_binary_metrics,
    get_amp_context,
    get_device,
    resolve_default_checkpoint,
)

LABEL_NAMES = {0: "negative", 1: "positive"}


def load_raw_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


@torch.no_grad()
def test_model(
    cfg: AppConfig = CFG,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    device = get_device(cfg.train.device)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else resolve_default_checkpoint(cfg.train.out_dir)
    )
    if output_dir is None:
        output_dir = checkpoint_path.parent / "test_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint.get("model_config", cfg.model.__dict__))
    model = DecoderSentimentClassifier(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    processed_dir = Path(cfg.data.processed_dir)
    test_ds = SentimentTensorDataset(processed_dir / "test.pt")
    raw_rows = load_raw_jsonl(processed_dir / "test.jsonl")

    loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_preds = []
    all_labels = []
    all_prob_pos = []
    weighted_loss_sum = 0.0
    total_examples = 0

    for batch in tqdm(loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with get_amp_context(device, cfg.train.mixed_precision):
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

        logits = out["logits"]
        loss = out["loss"]
        probs = torch.softmax(logits.float(), dim=-1)
        preds = probs.argmax(dim=-1)

        weighted_loss_sum += loss.detach().float().item() * labels.size(0)
        total_examples += labels.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_prob_pos.append(probs[:, 1].cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    prob_pos = torch.cat(all_prob_pos)
    avg_loss = weighted_loss_sum / max(1, total_examples)
    metrics = compute_binary_metrics(preds, labels, avg_loss)
    metrics["num_examples"] = int(total_examples)
    metrics["checkpoint"] = str(checkpoint_path)

    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_path = output_dir / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "label",
                "label_name",
                "prediction",
                "prediction_name",
                "prob_positive",
                "correct",
                "text",
            ],
        )
        writer.writeheader()
        for i, row in enumerate(raw_rows):
            label = int(labels[i].item())
            pred = int(preds[i].item())
            writer.writerow(
                {
                    "index": i,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "prediction": pred,
                    "prediction_name": LABEL_NAMES[pred],
                    "prob_positive": f"{float(prob_pos[i].item()):.6f}",
                    "correct": int(label == pred),
                    "text": row["text"].replace("\n", " "),
                }
            )

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test IMDb decoder sentiment classifier."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to runs/.../best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save test metrics/predictions.",
    )
    args = parser.parse_args()
    test_model(CFG, checkpoint_path=args.checkpoint, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
