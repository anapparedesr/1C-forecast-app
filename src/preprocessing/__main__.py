"""
src/preprocessing/__main__.py
------------------------------
Entry point for the preprocessing step.
Accepts CLI arguments for input/output paths.

Usage:
    python -m preprocessing \
        --raw-dir data/raw \
        --prep-dir data/prep \
        --inference-dir data/inference
"""

# -----
# Libraries
import argparse
from pathlib import Path

from prep import prepare_data
from prep_inference import prepare_inference


# -----
# Argument parser
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the preprocessing step.

    Returns:
    ---
    argparse.Namespace
        Parsed arguments with raw_dir, prep_dir, and inference_dir.
    """
    parser = argparse.ArgumentParser(
        description="Preprocessing step: prepare training and inference datasets."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--prep-dir",
        type=Path,
        default=Path("data/prep"),
        help="Directory where the prepared training dataset will be saved (default: data/prep)",
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=Path("data/inference"),
        help="Directory where the inference dataset will be saved (default: data/inference)",
    )
    return parser.parse_args()


# -----
# Entry point
if __name__ == "__main__":
    args = parse_args()
    prepare_data(raw_dir=args.raw_dir, prep_dir=args.prep_dir)
    prepare_inference(
        raw_dir=args.raw_dir,
        inference_dir=args.inference_dir,
        prep_dir=args.prep_dir,
    )
