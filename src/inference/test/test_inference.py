"""
src/inference/test/test_inference.py
--------------------------------------
Unit tests for the inference step functions.
"""

# -----
# Libraries
import pandas as pd
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import generate_predictions, save_predictions


# -----
# Fixtures
@pytest.fixture
def sample_inference_data():
    """Sample inference DataFrame with lag features."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "shop_id": [1, 1, 2],
            "item_id": [10, 20, 30],
            "lag_1": [2.0, 0.0, 1.0],
            "lag_3": [1.5, 0.0, 0.5],
            "lag_6": [1.0, 0.0, 0.0],
            "lag_12": [0.5, 0.0, 0.0],
        }
    )


@pytest.fixture
def mock_model():
    """Mock sklearn model that returns fixed predictions."""
    model = MagicMock()
    model.predict.return_value = [1.5, 0.0, 0.8]
    return model


# -----
# Tests for generate_predictions
def test_generate_predictions_adds_column(mock_model, sample_inference_data):
    """generate_predictions should add a 'prediction' column."""
    result = generate_predictions(mock_model, sample_inference_data)
    assert "prediction" in result.columns


def test_generate_predictions_correct_length(mock_model, sample_inference_data):
    """Output should have same number of rows as input."""
    result = generate_predictions(mock_model, sample_inference_data)
    assert len(result) == len(sample_inference_data)


def test_generate_predictions_does_not_modify_input(mock_model, sample_inference_data):
    """Original DataFrame should not be modified."""
    original_cols = list(sample_inference_data.columns)
    generate_predictions(mock_model, sample_inference_data)
    assert list(sample_inference_data.columns) == original_cols


def test_generate_predictions_calls_model(mock_model, sample_inference_data):
    """Model's predict method should be called once."""
    generate_predictions(mock_model, sample_inference_data)
    mock_model.predict.assert_called_once()


# -----
# Tests for save_predictions
def test_save_predictions_creates_file(mock_model, sample_inference_data, tmp_path):
    """save_predictions should create the output CSV file."""
    predictions = generate_predictions(mock_model, sample_inference_data)
    output_path = save_predictions(predictions, tmp_path)
    assert output_path.exists()


def test_save_predictions_correct_columns(mock_model, sample_inference_data, tmp_path):
    """Saved CSV should only contain ID and item_cnt_month columns."""
    predictions = generate_predictions(mock_model, sample_inference_data)
    output_path = save_predictions(predictions, tmp_path)
    saved = pd.read_csv(output_path)
    assert list(saved.columns) == ["ID", "item_cnt_month"]


def test_save_predictions_correct_row_count(mock_model, sample_inference_data, tmp_path):
    """Saved CSV should have the same number of rows as the input."""
    predictions = generate_predictions(mock_model, sample_inference_data)
    output_path = save_predictions(predictions, tmp_path)
    saved = pd.read_csv(output_path)
    assert len(saved) == len(sample_inference_data)
