"""Run the full IMDb sentiment classifier pipeline end-to-end."""

from __future__ import annotations

import argparse

from src.config import CFG
from src.prepare_dataset import prepare_dataset
from src.test import test_model
from src.train import train


def run_pipeline(force_prepare: bool = False, description: str = "") -> None:
    print("Step 1/3: Prepare dataset")
    prepare_dataset(CFG, force=force_prepare)

    print("\nStep 2/3: Train model")
    best_checkpoint = train(CFG, description=description)
    run_dir = best_checkpoint.parent

    print("\nStep 3/3: Test best checkpoint")
    test_model(
        CFG,
        checkpoint_path=best_checkpoint,
        output_dir=run_dir / "test_results",
    )

    print("\nDone.")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Run directory: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prepare -> train -> test end-to-end."
    )
    parser.add_argument(
        "--force-prepare", action="store_true", help="Rebuild tokenized dataset."
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Free-text run description saved only in training_summary.json.",
    )
    args = parser.parse_args()
    run_pipeline(force_prepare=args.force_prepare, description=args.description)


if __name__ == "__main__":
    main()
