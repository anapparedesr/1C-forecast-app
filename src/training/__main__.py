"""
src/training/__main__.py
-------------------------
Entry point for the training step.
Accepts CLI arguments for input/output paths and model hyperparameters.

Usage:
    python -m training \
        --prep-dir data/prep \
        --artifacts-dir artifacts \
        --n-estimators 200 \
        --max-depth 6 \
        --random-seed 42
"""

# -----
# Libraries
import argparse
from pathlib import Path

from train import train_and_evaluate


# -----
# Argument parser
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training step.

    Returns:
    ---
    argparse.Namespace
        Parsed arguments with prep_dir, artifacts_dir, and hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Training step: train a Random Forest model and save artifacts."
    )
    parser.add_argument(
        "--prep-dir",
        type=Path,
        default=Path("data/prep"),
        help="Directory containing the prepared dataset (default: data/prep)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where the trained model will be saved (default: artifacts)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum depth of each tree (default: 10)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=10,
        help="Random seed for reproducibility (default: 10)",
    )
    parser.add_argument(
        "--no-random-search",
        action="store_true",
        default=False,
        help="Disable RandomizedSearchCV and use default hyperparameters",
    )
    return parser.parse_args()


# -----
# Entry point
if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
    prep_dir=args.prep_dir,
    artifacts_dir=args.artifacts_dir,
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_seed=args.random_seed,
    use_random_search=not args.no_random_search,
)
