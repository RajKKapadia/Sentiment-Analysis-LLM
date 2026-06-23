"""Run inference with a trained decoder sentiment classifier.

Despite the filename, this model does classification, not text generation. The name is kept
because it mirrors your existing GPT project workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from src.config import CFG, AppConfig, ModelConfig
from src.create_model import DecoderSentimentClassifier
from src.train import get_amp_context, get_device, resolve_default_checkpoint

LABEL_NAMES = {0: "negative", 1: "positive"}


def encode_text(text: str, cfg: AppConfig) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tiktoken.get_encoding(cfg.data.tokenizer_name)
    token_ids = enc.encode_ordinary(text)
    token_ids = token_ids[: max(0, cfg.data.block_size - 1)] + [cfg.data.eos_token_id]

    if len(token_ids) < cfg.data.block_size:
        pad_len = cfg.data.block_size - len(token_ids)
        input_ids = token_ids + [cfg.data.pad_token_id] * pad_len
        attention_mask = [1] * len(token_ids) + [0] * pad_len
    else:
        input_ids = token_ids[: cfg.data.block_size]
        attention_mask = [1] * cfg.data.block_size

    return torch.tensor([input_ids], dtype=torch.long), torch.tensor(
        [attention_mask], dtype=torch.bool
    )


@torch.no_grad()
def predict_sentiment(
    text: str, cfg: AppConfig = CFG, checkpoint_path: str | Path | None = None
) -> dict:
    device = get_device(cfg.train.device)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else resolve_default_checkpoint(cfg.train.out_dir)
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint.get("model_config", cfg.model.__dict__))
    model = DecoderSentimentClassifier(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_ids, attention_mask = encode_text(text, cfg)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with get_amp_context(device, cfg.train.mixed_precision):
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = torch.softmax(out["logits"].float(), dim=-1)[0]
    pred = int(torch.argmax(probs).item())
    result = {
        "text": text,
        "prediction": pred,
        "sentiment": LABEL_NAMES[pred],
        "prob_negative": float(probs[0].item()),
        "prob_positive": float(probs[1].item()),
        "confidence": float(probs[pred].item()),
        "checkpoint": str(checkpoint_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer sentiment for custom text.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to runs/.../best.pt",
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Text/review to classify."
    )
    args = parser.parse_args()

    if args.text is None:
        args.text = input("Enter review text: ").strip()

    result = predict_sentiment(args.text, CFG, args.checkpoint)
    print(f"Sentiment: {result['sentiment']}  confidence={result['confidence']:.4f}")
    print(
        f"P(negative)={result['prob_negative']:.4f}  P(positive)={result['prob_positive']:.4f}"
    )


if __name__ == "__main__":
    main()
