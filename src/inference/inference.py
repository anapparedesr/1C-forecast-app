"""
inference.py
------
Execute inference with trained model using test with lags dataframe.

Pipeline:
    1. Load trained model
    2. Load inference data
    3. Generate predictions
    4. Save file with predictions
"""

# -----
# Libraries
import time
from pathlib import Path

import joblib
import pandas as pd

from utils.logger import setup_logger


# -----
# Constants
MODEL = "random_forest_lags.pkl"
TEST_WITH_LAGS = "test_with_lags.csv"
PREDICTIONS = "submissions.csv"

FEATURES_LAG = ["lag_1", "lag_3", "lag_6", "lag_12"]

logger = setup_logger("inference")


# -----
# Loading data and model function
def load_model(artifacts_dir: Path):
    """
    Load model

    Parameters:
    ---
    artifacts_dir: Path
        Directory with .pkl file

    Returns:
    ---
    object sklearn
        Model for inference
    """
    logger.info("Loading trained model...")
    try:
        model = joblib.load(artifacts_dir / MODEL)
        logger.info("Model loaded successfully: %s", MODEL)
        return model
    except FileNotFoundError:
        logger.error("Model file not found: %s", MODEL)
        raise

def load_inference_data(inference_dir: Path) -> pd.DataFrame:
    """
    Load inference data with lag features.

    Parameters
    ---
    inference_dir: Path
        Directory with test_with_lags.csv file.

    Returns
    ---
    pd.DataFrame
        DataFrame with lag columns ready for inference.
    """
    logger.info("Loading inference data...")
    inference_data = pd.read_csv(inference_dir / TEST_WITH_LAGS)
    logger.info("Inference data loaded: %s records", f"{len(inference_data):,}")
    return inference_data


# -----
# Prediction function
def generate_predictions(
    model,
    inference_data: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run the model on lag features and add 'prediction' column to the DataFrame.

    Parameters
    ---
    model: object sklearn
        Trained model with .predict() method.
    inference_data : pd.DataFrame
        DataFrame with lag features.
    features : list[str] | None
        Feature column names (default: FEATURES_LAG).

    Returns
    ---
    pd.DataFrame
        Original DataFrame with prediction column added.
    """
    if features is None:
        features = FEATURES_LAG

    logger.info("Generating predictions...")
    data_with_prediction = inference_data.copy()
    features_matrix = data_with_prediction[features]
    data_with_prediction["prediction"] = model.predict(features_matrix)
    logger.info(
        "Predictions generated: %s records", f"{len(data_with_prediction):,}"
    )
    return data_with_prediction


# -----
# Saving function
def save_predictions(
    predictions: pd.DataFrame,
    predictions_dir: Path,
) -> Path:
    """
    Save final predictions with only ID and item_cnt_month columns.

    Parameters
    ---
    predictions: pd.DataFrame
        DataFrame with predictions generated.
    predictions_dir : Path
        Output directory.

    Returns
    ---
    Path
        Full path of the saved file.
    """
    predictions_dir.mkdir(parents=True, exist_ok=True)
    exit_path = predictions_dir / PREDICTIONS

    output = predictions[["ID", "prediction"]].rename(
        columns={"prediction": "item_cnt_month"}
    )
    output.to_csv(exit_path, index=False)
    logger.info(
        "Predictions saved: %s records → %s", f"{len(output):,}", PREDICTIONS
    )
    return exit_path


# -----
# Main function
def execute_inference(
    inference_dir: Path,
    artifacts_dir: Path,
    predictions_dir: Path,
) -> None:
    """
    Execute the full inference pipeline:
    1. Load model.
    2. Load inference data.
    3. Generate predictions.
    4. Save predictions.

    Parameters
    ---
    inference_dir: Path
        Directory with inference data (test with lags).
    artifacts_dir: Path
        Directory with trained model.
    predictions_dir: Path
        Directory for saving predictions.
    """
    logger.info("=" * 50)
    logger.info("Starting inference pipeline (inference.py)")
    logger.info("=" * 50)
    start_time = time.time()

    try:
        model = load_model(artifacts_dir)
        inference_data = load_inference_data(inference_dir)
        predictions = generate_predictions(model, inference_data)
        save_predictions(predictions, predictions_dir)

    except FileNotFoundError as e:
        logger.error("Input file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during inference: %s", e)
        raise

    duration = time.time() - start_time
    logger.info("Inference pipeline completed in %.2f seconds", duration)
    logger.info("=" * 50)


# -----
# Entry point
if __name__ == "__main__":
    BASE_PATH = Path(__file__).resolve().parent.parent
    execute_inference(
        inference_dir=BASE_PATH / "data" / "inference",
        artifacts_dir=BASE_PATH / "artifacts",
        predictions_dir=BASE_PATH / "data" / "predictions",
    )
