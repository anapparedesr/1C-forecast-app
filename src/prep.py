"""
prep.py
-----
Module for data preparation for sales forecasting.

Main steps:
    1. Load and clean sales data.
    2. Aggregate data at monthly level by store and product.
    3. Build the complete grid of combinations.
    4. Generate lags (1,3,6 and 12 months) of the target variable.
    5. Save the prepared dataset for modeling.

"""

# -----
# Libraries
from pathlib import Path
import time
import pandas as pd
from src.utils.logger import setup_logger


# -----
# Constants
SALES_RAW = "sales_train.csv"
ITEMS_RAW = "items.csv"
GRID_MODEL_PREP = "grid_model.csv"

COLUMN_PRICE = "item_price"
COLUMN_UNITS = "item_cnt_day"
COLUMN_UNITS_MONTH = "item_cnt_month"

CLIP = 20
LAGS = [1, 3, 6, 12]

logger = setup_logger("prep")

# -----
# Data loading functions
def load_sales(raw_dir: Path) -> pd.DataFrame:
    """Load raw sales data from the specified directory.

    Parameters:
    ---
    raw_dir: Path
        Directory containing raw data files (csv files)

    Returns:
    ---
    pd.DataFrame
        DataFrame containing the loaded data (sales_train.csv)
    """
    logger.info("Loading sales data...")
    sales = pd.read_csv(raw_dir / SALES_RAW)
    logger.info("Sales data loaded: %s records", f"{len(sales):,}")

    if sales[COLUMN_PRICE].isna().sum() > 0:
        logger.warning(
            "Found %s missing values in '%s'",
            f"{sales[COLUMN_PRICE].isna().sum():,}",
            COLUMN_PRICE,
        )
    if sales[COLUMN_UNITS].isna().sum() > 0:
        logger.warning(
            "Found %s missing values in '%s'",
            f"{sales[COLUMN_UNITS].isna().sum():,}",
            COLUMN_UNITS,
        )
    return sales

def load_items(raw_dir: Path) -> pd.DataFrame:
    """Load items metadata from the specified directory.

    Parameters:
    ---
    raw_dir: Path
        Directory containing raw data files (csv files)

    Returns:
    ---
    pd.DataFrame
        DataFrame containing the loaded data (items.csv)
    """
    logger.info("Loading items data...")
    items = pd.read_csv(raw_dir / ITEMS_RAW)
    logger.info("Items data loaded: %s records", f"{len(items):,}")
    return items

# -----
# Cleaning functions
def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Delete rows with non-positive price and negative units.

    Parameters:
    ---
    sales: pd.DataFrame
        DataFrame containing the sales data to be cleaned.

    Returns:
    ---
    pd.DataFrame
        Cleaned DataFrame with only valid price and units.
    """
    logger.info("Cleaning sales data...")
    has_valid_price = sales[COLUMN_PRICE] > 0
    has_valid_units = sales[COLUMN_UNITS] >= 0
    sales_cleaned = sales[has_valid_price & has_valid_units].copy()

    removed = len(sales) - len(sales_cleaned)
    logger.info(
        "Removed %s invalid records — %s remaining",
        f"{removed:,}",
        f"{len(sales_cleaned):,}",
    )
    return sales_cleaned


# -----
# Aggregation and enrichment functions
def aggregate_monthly(sales_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales data to monthly level by store and product,
    then clip the target variable to reduce the impact of outliers.

    Parameters:
    ---
    sales_cleaned: pd.DataFrame
        Cleaned DataFrame containing the sales data.

    Returns:
    ---
    pd.DataFrame
        Aggregated DataFrame with columns: date_block_num, shop_id, item_id, item_cnt_month.
    """
    logger.info("Aggregating sales to monthly level...")
    monthly_sales = (
        sales_cleaned.groupby(
            ["date_block_num", "shop_id", "item_id"], as_index=False
        )
        .agg({COLUMN_UNITS: "sum"})
        .rename(columns={COLUMN_UNITS: COLUMN_UNITS_MONTH})
    )

    monthly_sales[COLUMN_UNITS_MONTH] = monthly_sales[COLUMN_UNITS_MONTH].clip(0, CLIP)
    logger.info(
        "Monthly aggregation complete: %s records — clip applied at %s",
        f"{len(monthly_sales):,}",
        CLIP,
    )
    return monthly_sales


def add_item_category(
    monthly_sales: pd.DataFrame, items: pd.DataFrame
) -> pd.DataFrame:
    """
    Add item category information to the monthly sales data.

    Parameters:
    ---
    monthly_sales: pd.DataFrame
        DataFrame containing the monthly sales data
    items: pd.DataFrame
        DataFrame containing the item information (item_id, item_category_id)

    Returns:
    ---
    pd.DataFrame
        DataFrame with the item category information added.
    """
    logger.info("Adding item category information...")
    enriched = monthly_sales.merge(
        items[["item_id", "item_category_id"]], on="item_id", how="left"
    )
    missing_category = enriched["item_category_id"].isna().sum()
    if missing_category > 0:
        logger.warning(
            "Found %s records without item_category_id after merge",
            f"{missing_category:,}",
        )
    return enriched


# -----
# Grid construction and lag generation functions
def build_grid(monthly_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete grid of active combinations of shop_id and item_id,
    then join with the monthly sales data to fill in the target variable.
    All combinations without sales data will have item_cnt_month set to 0.

    Parameters:
    ---
    monthly_sales: pd.DataFrame
        DataFrame containing the monthly sales data with item category information.

    Returns:
    ---
    pd.DataFrame
        Complete grid with columns: date_block_num, shop_id, item_id, item_cnt_month.
    """
    logger.info("Building complete shop-item-month grid...")
    monthly_sales_sorted = monthly_sales.sort_values(
        ["shop_id", "item_id", "date_block_num"]
    )
    total_months = sorted(monthly_sales_sorted["date_block_num"].unique())
    comb_active = monthly_sales_sorted[["shop_id", "item_id"]].drop_duplicates()
    logger.info(
        "Grid dimensions: %s months x %s active combinations",
        len(total_months),
        f"{len(comb_active):,}",
    )

    grid_list = [
        comb_active.assign(date_block_num=month) for month in total_months
    ]
    grid = pd.concat(grid_list, ignore_index=True)[
        ["date_block_num", "shop_id", "item_id"]
    ]

    grid = grid.merge(
        monthly_sales_sorted[
            ["date_block_num", "shop_id", "item_id", COLUMN_UNITS_MONTH]
        ],
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )
    grid[COLUMN_UNITS_MONTH] = grid[COLUMN_UNITS_MONTH].fillna(0)
    logger.info("Grid built: %s total records", f"{len(grid):,}")
    return grid.sort_values(["shop_id", "item_id", "date_block_num"])

def add_lags(grid: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """
    Generate lag columns for the target variable (item_cnt_month)
    grouped by shop_id and item_id.

    Parameters:
    ---
    grid: pd.DataFrame
        DataFrame containing the complete grid with the target variable.
    lags: list[int] | None
        List of lag periods to generate (default: [1, 3, 6, 12])

    Returns:
    ---
    pd.DataFrame
        DataFrame with the lag columns added for the target variable.
    """
    if lags is None:
        lags = LAGS

    logger.info("Generating lag features: %s...", lags)
    grid_with_lags = grid.copy()
    for lag in lags:
        grid_with_lags[f"lag_{lag}"] = (
            grid_with_lags.groupby(["shop_id", "item_id"])[COLUMN_UNITS_MONTH].shift(
                lag
            )
        )
    logger.info("Lag features generated successfully")
    return grid_with_lags

def filter_complete_cases(
    grid_with_lags: pd.DataFrame, lags: list[int] | None = None
) -> pd.DataFrame:
    """
    Drop rows with missing values in the lag columns to keep only complete cases.
    (first months will have missing values due to lag generation)

    Parameters:
    ---
    grid_with_lags: pd.DataFrame
        DataFrame containing the grid with lag columns added.
    lags: list[int] | None
        List of lag periods that were generated (default: [1, 3, 6, 12])

    Returns:
    ---
    pd.DataFrame
        DataFrame with incomplete rows dropped, ready for modeling.
    """
    if lags is None:
        lags = LAGS

    lag_columns = [f"lag_{lag}" for lag in lags]
    grid_model = grid_with_lags.dropna(subset=lag_columns)
    dropped = len(grid_with_lags) - len(grid_model)
    logger.info(
        "Filtered incomplete cases: %s rows dropped — %s remaining",
        f"{dropped:,}",
        f"{len(grid_model):,}",
    )
    return grid_model

# -----
# Principal function to prepare the data for modeling
def prepare_data(raw_dir: Path, prep_dir: Path) -> None:
    """
    Defined the main function to prepare the data for modeling. It executes the following steps:
    1. Load raw data (sales and items).
    2. Clean the sales data by removing invalid price and units.
    3. Aggregate the cleaned sales data to monthly level by store and product.
    4. Add item category information by merging with the items DataFrame.
    5. Build the complete grid of combinations and fill in the target variable.
    6. Generate lag columns for the target variable.
    7. Filter to keep only complete cases for modeling.
    8. Save the prepared dataset to the specified directory.

    Parameters:
    ---
    raw_dir: Path
        Directory containing the raw data files (sales_train.csv and items.csv).
    prep_dir: Path
        Directory where the prepared dataset (grid_model.csv) will be saved.
    """

    logger.info("=" * 50)
    logger.info("Starting data preparation pipeline (prep.py)")
    logger.info("=" * 50)
    start_time = time.time()

    try:
        sales_raw = load_sales(raw_dir)
        items = load_items(raw_dir)

        sales_cleaned = clean_sales(sales_raw)
        monthly_sales = aggregate_monthly(sales_cleaned)
        monthly_sales = add_item_category(monthly_sales, items)

        grid = build_grid(monthly_sales)
        grid_with_lags = add_lags(grid)
        grid_model = filter_complete_cases(grid_with_lags)

        prep_dir.mkdir(parents=True, exist_ok=True)
        grid_model.to_csv(prep_dir / GRID_MODEL_PREP, index=False)
        logger.info(
            "Prepared dataset saved: %s records → %s",
            f"{len(grid_model):,}",
            GRID_MODEL_PREP,
        )

    except FileNotFoundError as e:
        logger.error("Input file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during data preparation: %s", e)
        raise

    duration = time.time() - start_time
    logger.info("Data preparation completed in %.2f seconds", duration)
    logger.info("=" * 50)

# -----
# Entry point
if __name__ == "__main__":
    BASE_PATH = Path("data")
    prepare_data(
        raw_dir=BASE_PATH / "raw",
        prep_dir=BASE_PATH / "prep",
    )
