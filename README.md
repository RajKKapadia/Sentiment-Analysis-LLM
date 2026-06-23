# Decoder-only IMDb Sentiment Classifier

This project trains a GPT-style decoder Transformer for binary sentiment prediction on IMDb movie reviews.

## Files

- `config.py` - model, data, and training configuration
- `prepare_dataset.py` - downloads IMDb, tokenizes with `tiktoken` GPT-2 BPE, creates train/valid/test splits
- `create_model.py` - decoder-only Transformer backbone plus classification head
- `train.py` - trains, validates, saves checkpoints and metrics
- `test.py` - evaluates best checkpoint on test set, saves metrics and predictions
- `main.py` - runs prepare -> train -> test end-to-end
- `generate.py` - runs sentiment inference for custom text

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fast smoke test

Edit `config.py`:

```python
max_samples = 2000
```

Then run:

```bash
python main.py --force-prepare
```

## Full run

Set `max_samples = None` in `config.py`, then:

```bash
python main.py --force-prepare --description "baseline full IMDb run"
```

Or run steps manually:

```bash
python prepare_dataset.py --force
python train.py --description "baseline full IMDb run"
python test.py
python generate.py --text "This movie was surprisingly beautiful and emotional."
```

## Outputs

Each training run creates a timestamped folder under `runs/imdb_decoder_classifier/`.
The timestamp format is `HH_MM_DD_MM_YYYY`.

```text
runs/imdb_decoder_classifier/
└── HH_MM_DD_MM_YYYY/
    ├── best.pt
    ├── latest.pt
    ├── metrics.jsonl
    ├── training_summary.json
    ├── config_snapshot.json
    └── test_results/
        ├── test_metrics.json
        └── test_predictions.csv
```

`test.py` and `generate.py` default to the latest timestamped run when no
`--checkpoint` is provided.

Use `--description` on `main.py` or `train.py` to save free-text run context in
`training_summary.json` only.

## Architecture

```text
input_ids [B, T]
  -> token embedding + position embedding
  -> N causal decoder blocks
  -> final layer norm
  -> last non-padding token hidden state [B, C]
  -> linear classification head [B, 2]
```

The last non-padding token is used because a causal decoder's final useful token can attend to the full review prefix.
