"""
src/utils/metrics.py
---------------------
Reusable functions to calculate and report evaluation metrics.
"""

# -----
# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


# -----
# Calculare RMSE
def calculate_rmse(y_real: pd.Series, y_predicted: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_real : pd.Series
        Real target values.
    y_predicted : np.ndarray
        Model predictions.

    Returns
    -------
    float
        RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_real, y_predicted)))


# -----
# Calculate MAE
def calculate_mae(y_real: pd.Series, y_predicted: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE).

    Parameters
    ----------
    y_real : pd.Series
        Real target values.
    y_predicted : np.ndarray
        Model predictions.

    Returns
    -------
    float
        MAE value.
    """
    return float(mean_absolute_error(y_real, y_predicted))


# -----
# Report metrics
def report_metrics(
    y_real: pd.Series, y_predicted: np.ndarray, model_name: str = "Model"
) -> dict[str, float]:
    """
    Calculate RMSE and MAE and print a formatted summary.

    Parameters
    ----------
    y_real : pd.Series
        Real target values.
    y_predicted : np.ndarray
        Model predictions.
    model_name : str
        Descriptive label for the report.

    Returns
    -------
    dict[str, float]
        Dictionary with calculated metrics: {'rmse': ..., 'mae': ...}.
    """
    rmse = calculate_rmse(y_real, y_predicted)
    mae = calculate_mae(y_real, y_predicted)

    print(f"[{model_name}] RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return {"rmse": rmse, "mae": mae}
