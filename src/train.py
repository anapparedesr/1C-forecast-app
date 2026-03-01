"""
train.py
-----
Module for training a Random Forest model for sales forecasting using the prepared dataset.
Evaluates the model on a validation set and saves the trained model for future use.

Main steps:
    1. Load the prepared dataset (grid_model.csv).
    2. Split the data into training and validation sets based on date_block_num.
    3. Train a Random Forest regressor using lag features as predictors.
    4. Evaluate the model using RMSE on the validation set.
    5. Save the trained model to the artifacts directory.
"""

# -----
# Libraries
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.utils.logger import setup_logger


# -----
# Constants
GRID_MODEL_PREP = "grid_model.csv"
MODEL_PATH = "random_forest_lags.pkl"

FEATURE_LAGS = ["lag_1", "lag_3", "lag_6", "lag_12"]
TARGET_COL = "item_cnt_month"

MONTH_TRAIN = 32
MONTH_VAL = 33
CLIP = 20

RANDOM_SEED = 10
N_ESTIMATORS = 100
MAX_DEPTH = 10

logger = setup_logger("train")

# -----
# Loading data function
def load_prepared_data(prep_dir: Path) -> pd.DataFrame:
    """
    Load the prepared dataset for modeling.

    Parameters:
    ---
    prep_dir: Path
        Directory containing the prepared dataset (grid_model.csv)

    Returns:
    ---
    pd.DataFrame
        DataFrame with lag features and target variable.
    """
    logger.info("Loading prepared dataset...")
    grid = pd.read_csv(prep_dir / GRID_MODEL_PREP)
    logger.info("Prepared dataset loaded: %s records", f"{len(grid):,}")
    return grid

# -----
# Splitting data function
def split_data(
    grid_model: pd.DataFrame,
    month_train: int = MONTH_TRAIN,
    month_val: int = MONTH_VAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the prepared dataset into training and validation sets based on date_block_num.

    Parameters:
    ---
    grid_model: pd.DataFrame
        DataFrame with lag features and target variable.
    month_train: int
        Last month to include in the training set (inclusive).
    month_val: int
        Month to use as the validation set.

    Returns:
    ---
    tuple[pd.DataFrame, pd.DataFrame]
        (training_grid, validation_grid)
    """
    train_grid = grid_model[grid_model["date_block_num"] <= month_train].copy()
    val_grid = grid_model[grid_model["date_block_num"] == month_val].copy()
    logger.info(
        "Train set: %s records (months 0–%s)", f"{len(train_grid):,}", month_train
    )
    logger.info(
        "Validation set: %s records (month %s)", f"{len(val_grid):,}", month_val
    )
    return train_grid, val_grid

def features_and_target(
    grid: pd.DataFrame,
    features: list[str] | None = None,
    clip: float = CLIP,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable, applying clipping to the target.

    Parameters:
    ---
    grid: pd.DataFrame
        Subset of the prepared dataset containing lag features and target variable.
    features: list[str] | None
        List of feature column names (default: FEATURE_LAGS).
    clip: float
        Maximum value to clip the target variable to prevent extreme values.

    Returns:
    ---
    tuple[pd.DataFrame, pd.Series]
        (features_matrix, target)
    """
    if features is None:
        features = FEATURE_LAGS

    features_matrix = grid[features]
    target = grid[TARGET_COL].clip(0, clip)
    return features_matrix, target

# -----
# Training and evaluation function
def train_model(
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> RandomForestRegressor:
    """
    Train a Random Forest regressor.

    Parameters:
    ---
    features_train: pd.DataFrame
        Training features (lag columns).
    target_train: pd.Series
        Training target variable.

    Returns:
    ---
    RandomForestRegressor
        Trained model.
    """
    logger.info(
        "Training Random Forest — n_estimators=%s, max_depth=%s, random_state=%s",
        N_ESTIMATORS,
        MAX_DEPTH,
        RANDOM_SEED,
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(features_train, target_train)
    logger.info("Model training complete")
    return model

def calculate_rmse(
    model: RandomForestRegressor,
    features_val: pd.DataFrame,
    target_val: pd.Series,
) -> float:
    """
    Calculate the RMSE of the model on the validation set.

    Parameters:
    ---
    model: RandomForestRegressor
        Trained model.
    features_val: pd.DataFrame
        Validation features.
    target_val: pd.Series
        Validation target variable.

    Returns:
    ---
    float
        RMSE value.
    """
    predictions = model.predict(features_val)
    rmse = float(np.sqrt(mean_squared_error(target_val, predictions)))
    logger.info("Validation RMSE (Random Forest): %.4f", rmse)
    return rmse


# -----
# Saving model function
def save_model(model: RandomForestRegressor, artifacts_dir: Path) -> Path:
    """
    Save the trained model to the artifacts directory.

    Parameters:
    ---
    model: RandomForestRegressor
        Trained model to be saved.
    artifacts_dir: Path
        Directory where the model will be saved.

    Returns:
    ---
    Path
        Path to the saved model file.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / MODEL_PATH
    joblib.dump(model, model_path)
    logger.info("Model saved: %s", MODEL_PATH)
    return model_path

# -----
# Main function to execute the training process
def train_and_evaluate(prep_dir: Path, artifacts_dir: Path) -> float:
    """
    Execute the full training and evaluation pipeline:
    1. Load the prepared dataset.
    2. Split into training and validation sets.
    3. Extract features and target variable.
    4. Train the Random Forest model.
    5. Calculate RMSE on the validation set.
    6. Save the trained model.

    Parameters:
    ---
    prep_dir: Path
        Directory containing the prepared dataset (grid_model.csv).
    artifacts_dir: Path
        Directory where the trained model will be saved.

    Returns:
    ---
    float
        RMSE value on the validation set.
    """
    logger.info("=" * 50)
    logger.info("Starting training pipeline (train.py)")
    logger.info("=" * 50)
    start_time = time.time()

    try:
        grid = load_prepared_data(prep_dir)
        train_grid, val_grid = split_data(grid)

        features_train, target_train = features_and_target(train_grid)
        features_val, target_val = features_and_target(val_grid)

        model = train_model(features_train, target_train)
        rmse = calculate_rmse(model, features_val, target_val)

        save_model(model, artifacts_dir)

    except FileNotFoundError as e:
        logger.error("Input file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during training: %s", e)
        raise

    duration = time.time() - start_time
    logger.info("Training pipeline completed in %.2f seconds", duration)
    logger.info("=" * 50)
    return rmse

# -----
# Entry point
if __name__ == "__main__":
    BASE_PATH = Path(__file__).resolve().parent.parent
    train_and_evaluate(
        prep_dir=BASE_PATH / "data" / "prep",
        artifacts_dir=BASE_PATH / "artifacts",
    )
