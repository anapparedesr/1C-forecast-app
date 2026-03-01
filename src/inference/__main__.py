"""
src/inference/__main__.py
--------------------------
Entry point for the inference step.
Accepts CLI arguments for input/output paths.

Usage:
    python -m inference \
        --inference-dir data/inference \
        --artifacts-dir artifacts \
        --predictions-dir data/predictions
"""

# -----
# Libraries
import argparse
from pathlib import Path

from inference import execute_inference


# -----
# Argument parser
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the inference step.

    Returns:
    ---
    argparse.Namespace
        Parsed arguments with inference_dir, artifacts_dir, and predictions_dir.
    """
    parser = argparse.ArgumentParser(
        description="Inference step: generate predictions using the trained model."
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=Path("data/inference"),
        help="Directory with test_with_lags.csv (default: data/inference)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing the trained model (default: artifacts)",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("data/predictions"),
        help="Directory where predictions will be saved (default: data/predictions)",
    )
    return parser.parse_args()


# -----
# Entry point
if __name__ == "__main__":
    args = parse_args()
    execute_inference(
        inference_dir=args.inference_dir,
        artifacts_dir=args.artifacts_dir,
        predictions_dir=args.predictions_dir,
    )
