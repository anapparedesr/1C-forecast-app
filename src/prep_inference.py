"""
prep_inference.py
-----
Module for preparing the inference dataset for sales forecasting.

Because the test set belongs to the month 34, we need to generate the lag features for that month
based on the previous months' data.
Lags are generated following the next logic:
- lag_1 -> month 33
- lag_3 -> month 31
- lag_6 -> month 28
- lag_12 -> month 22
Missing values are filled with 0
"""

# ----
# Libraries
from pathlib import Path
import time
import pandas as pd
from src.utils.logger import setup_logger

# ----
# Constants
TEST_RAW = "test.csv"
GRID_MODEL_PREP = "grid_model.csv"
TEST_WITH_LAGS = "test_with_lags.csv"

PREDICTION_MONTH = 34
MONTHS_PER_LAG = {"lag_1": 33, "lag_3": 31, "lag_6": 28, "lag_12": 22}

logger = setup_logger("prep_inference")

# ----
# Data loading functions
def load_test(raw_dir: Path) -> pd.DataFrame:
    """Load raw test data from the specified directory.

    Parameters:
    ---
    raw_dir: Path
        Directory containing raw data files (csv files)

    Returns:
    ---
    pd.DataFrame
        DataFrame with the combinations of store_id and item_id for prediction month.
    """
    logger.info("Loading test data...")
    test = pd.read_csv(raw_dir / TEST_RAW)
    logger.info("Test data loaded: %s records", f"{len(test):,}")
    return test

def load_grid_model(prep_dir: Path) -> pd.DataFrame:
    """
    Load the prepared dataset for modeling.

    Parameters:
    ---
    prep_dir: Path
        Directory containing the prepared dataset (grid_model.csv)

    Returns:
    ---
    pd.DataFrame
        DataFrame with columns: date_block_num, shop_id, item_id, item_cnt_month.
    """
    logger.info("Loading prepared grid model...")
    grid_model = pd.read_csv(prep_dir / GRID_MODEL_PREP)
    logger.info("Grid model loaded: %s records", f"{len(grid_model):,}")
    return grid_model

# -----
# Lag extraction function
def extract_monthly_lag(
    grid_model: pd.DataFrame,
    lag_name: str,
    origin_month: int,
) -> pd.DataFrame:
    """
    Extract sales from a specific historical month and rename as the lag column.

    Parameters
    ---
    grid_model: pd.DataFrame
        Training grid with historical sales.
    lag_name : str
        Lag column name (e.g. 'lag_1').
    origin_month: int
        date_block_num value to extract sales from.

    Returns
    ---
    pd.DataFrame
        DataFrame with columns: shop_id, item_id, <lag_name>.
    """
    return (
        grid_model[grid_model["date_block_num"] == origin_month][
            ["shop_id", "item_id", "item_cnt_month"]
        ].rename(columns={"item_cnt_month": lag_name})
    )

# -----
# Features creating function
def merge_lags_with_test(
    test: pd.DataFrame,
    grid_model: pd.DataFrame,
    months_per_lag: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Merge lag features into the test DataFrame using shop_id and item_id.

    Parameters
    ---
    test: pd.DataFrame
        Raw test DataFrame.
    grid_model: pd.DataFrame
        Grid with monthly historical sales.
    months_per_lag: dict[str, int] | None
        Mapping of lag name to origin month (default: MONTHS_PER_LAG).

    Returns
    ---
    pd.DataFrame
        Test DataFrame enriched with lag columns.
    """
    if months_per_lag is None:
        months_per_lag = MONTHS_PER_LAG

    logger.info("Merging lag features into test data...")
    test_with_lags = test.copy()

    for lag_name, origin_month in months_per_lag.items():
        lag_features = extract_monthly_lag(grid_model, lag_name, origin_month)
        test_with_lags = test_with_lags.merge(
            lag_features, on=["shop_id", "item_id"], how="left"
        )
        missing = test_with_lags[lag_name].isna().sum()
        if missing > 0:
            logger.warning(
                "%s: %s records without historical data — will be filled with 0",
                lag_name,
                f"{missing:,}",
            )

    lag_columns = list(months_per_lag.keys())
    test_with_lags[lag_columns] = test_with_lags[lag_columns].fillna(0)
    logger.info("All lag features merged and null values filled with 0")
    return test_with_lags

# -----
# Main function
def prepare_inference(
    raw_dir: Path,
    inference_dir: Path,
    prep_dir: Path,
) -> None:
    """
    Execute the full inference data preparation pipeline:
    1. Load test data and prepared grid.
    2. Merge lag features into the test set.
    3. Save the enriched test set ready for predictions.

    Parameters
    ---
    raw_dir : Path
        Directory with raw data.
    inference_dir : Path
        Directory where test with lags will be saved.
    prep_dir : Path
        Directory with the prepared training grid.
    """

    logger.info("=" * 50)
    logger.info("Starting inference data preparation pipeline (prep_inference.py)")
    logger.info("=" * 50)
    start_time = time.time()

    try:
        test = load_test(raw_dir)
        grid_model = load_grid_model(prep_dir)

        test_with_lags = merge_lags_with_test(test, grid_model)

        inference_dir.mkdir(parents=True, exist_ok=True)
        test_with_lags.to_csv(inference_dir / TEST_WITH_LAGS, index=False)
        logger.info(
            "Test with lags saved: %s records → %s",
            f"{len(test_with_lags):,}",
            TEST_WITH_LAGS,
        )

    except FileNotFoundError as e:
        logger.error("Input file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during inference preparation: %s", e)
        raise

    duration = time.time() - start_time
    logger.info("Inference preparation completed in %.2f seconds", duration)
    logger.info("=" * 50)

# -----
# Entry point
if __name__ == "__main__":
    BASE_PATH = Path("data")
    prepare_inference(
        raw_dir=BASE_PATH / "raw",
        inference_dir=BASE_PATH / "inference",
        prep_dir=BASE_PATH / "prep",
    )
