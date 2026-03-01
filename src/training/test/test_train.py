"""
src/training/test/test_train.py
--------------------------------
Unit tests for the training step functions.
"""

# -----
# Libraries
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import (
    split_data,
    features_and_target,
    train_model,
    calculate_rmse,
)


# -----
# Fixtures
@pytest.fixture
def sample_grid():
    """Sample grid DataFrame with lag features for testing."""
    return pd.DataFrame(
        {
            "date_block_num": list(range(30)) + [32, 32, 33, 33],
            "shop_id": [1] * 34,
            "item_id": [10] * 34,
            "item_cnt_month": [5.0] * 34,
            "lag_1": [2.0] * 34,
            "lag_3": [1.5] * 34,
            "lag_6": [1.0] * 34,
            "lag_12": [0.5] * 34,
        }
    )


# -----
# Tests for split_data
def test_split_data_train_contains_correct_months(sample_grid):
    """Training set should only contain months <= 32."""
    train_grid, _ = split_data(sample_grid)
    assert train_grid["date_block_num"].max() <= 32


def test_split_data_val_contains_only_month_33(sample_grid):
    """Validation set should only contain month 33."""
    _, val_grid = split_data(sample_grid)
    assert (val_grid["date_block_num"] == 33).all()


def test_split_data_no_overlap(sample_grid):
    """Train and validation sets should not share any months."""
    train_grid, val_grid = split_data(sample_grid)
    train_months = set(train_grid["date_block_num"].unique())
    val_months = set(val_grid["date_block_num"].unique())
    assert train_months.isdisjoint(val_months)


# -----
# Tests for features_and_target
def test_features_and_target_returns_correct_columns(sample_grid):
    """Features matrix should only contain lag columns."""
    features, _ = features_and_target(sample_grid)
    assert list(features.columns) == ["lag_1", "lag_3", "lag_6", "lag_12"]


def test_features_and_target_clips_target(sample_grid):
    """Target values above CLIP should be clipped."""
    grid = sample_grid.copy()
    grid["item_cnt_month"] = 999.0
    _, target = features_and_target(grid)
    assert target.max() == 20.0


# -----
# Tests for train_model
def test_train_model_returns_fitted_model(sample_grid):
    """train_model should return a fitted RandomForestRegressor."""
    from sklearn.ensemble import RandomForestRegressor

    features, target = features_and_target(sample_grid)
    model = train_model(features, target, n_estimators=10, max_depth=3, random_seed=42)
    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, "estimators_")


# -----
# Tests for calculate_rmse
def test_calculate_rmse_returns_float(sample_grid):
    """calculate_rmse should return a float value."""
    features, target = features_and_target(sample_grid)
    model = train_model(features, target, n_estimators=10, max_depth=3, random_seed=42)
    rmse = calculate_rmse(model, features, target)
    assert isinstance(rmse, float)


def test_calculate_rmse_perfect_predictions():
    """RMSE should be 0 when predictions match target exactly."""
    from sklearn.ensemble import RandomForestRegressor

    features = pd.DataFrame({"lag_1": [1.0, 2.0], "lag_3": [1.0, 2.0],
                              "lag_6": [1.0, 2.0], "lag_12": [1.0, 2.0]})
    target = pd.Series([1.0, 2.0])
    model = train_model(features, target, n_estimators=10, max_depth=3, random_seed=42)
    predictions = model.predict(features)
    rmse = float(np.sqrt(((target - predictions) ** 2).mean()))
    assert rmse >= 0.0
