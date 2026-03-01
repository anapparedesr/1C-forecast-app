"""
src/preprocessing/test/test_prep.py
------------------------------------
Unit tests for the preprocessing step functions.
"""

# -----
# Libraries
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prep import (
    clean_sales,
    aggregate_monthly,
    filter_complete_cases,
    add_item_category,
)


# -----
# Fixtures
@pytest.fixture
def sample_sales():
    """Sample sales DataFrame for testing."""
    return pd.DataFrame(
        {
            "date_block_num": [0, 0, 0, 1],
            "shop_id": [1, 1, 2, 1],
            "item_id": [10, 10, 20, 10],
            "item_price": [100.0, 50.0, -10.0, 200.0],
            "item_cnt_day": [2.0, 3.0, 1.0, -1.0],
        }
    )


@pytest.fixture
def sample_items():
    """Sample items DataFrame for testing."""
    return pd.DataFrame(
        {
            "item_id": [10, 20, 30],
            "item_category_id": [1, 2, 3],
        }
    )


# -----
# Tests for clean_sales
def test_clean_sales_removes_negative_price(sample_sales):
    """Rows with negative or zero price should be removed."""
    result = clean_sales(sample_sales)
    assert (result["item_price"] > 0).all()


def test_clean_sales_removes_negative_units(sample_sales):
    """Rows with negative units should be removed."""
    result = clean_sales(sample_sales)
    assert (result["item_cnt_day"] >= 0).all()


def test_clean_sales_keeps_valid_rows(sample_sales):
    """Only rows with price > 0 and units >= 0 should remain."""
    result = clean_sales(sample_sales)
    assert len(result) == 2


# -----
# Tests for aggregate_monthly
def test_aggregate_monthly_sums_units():
    """Daily sales for the same shop/item/month should be summed."""
    sales = pd.DataFrame(
        {
            "date_block_num": [0, 0],
            "shop_id": [1, 1],
            "item_id": [10, 10],
            "item_cnt_day": [2.0, 3.0],
        }
    )
    result = aggregate_monthly(sales)
    assert result["item_cnt_month"].iloc[0] == 5.0


def test_aggregate_monthly_clips_at_20():
    """Values above CLIP (20) should be clipped."""
    sales = pd.DataFrame(
        {
            "date_block_num": [0],
            "shop_id": [1],
            "item_id": [10],
            "item_cnt_day": [999.0],
        }
    )
    result = aggregate_monthly(sales)
    assert result["item_cnt_month"].iloc[0] == 20.0


# -----
# Tests for add_item_category
def test_add_item_category_adds_column(sample_items):
    """item_category_id column should be added after merge."""
    monthly_sales = pd.DataFrame(
        {
            "date_block_num": [0],
            "shop_id": [1],
            "item_id": [10],
            "item_cnt_month": [5.0],
        }
    )
    result = add_item_category(monthly_sales, sample_items)
    assert "item_category_id" in result.columns


def test_add_item_category_correct_value(sample_items):
    """item_category_id should match the value from items DataFrame."""
    monthly_sales = pd.DataFrame(
        {
            "date_block_num": [0],
            "shop_id": [1],
            "item_id": [10],
            "item_cnt_month": [5.0],
        }
    )
    result = add_item_category(monthly_sales, sample_items)
    assert result["item_category_id"].iloc[0] == 1


# -----
# Tests for filter_complete_cases
def test_filter_complete_cases_drops_nulls():
    """Rows with NaN in lag columns should be dropped."""
    grid = pd.DataFrame(
        {
            "shop_id": [1, 1],
            "item_id": [10, 10],
            "item_cnt_month": [5.0, 3.0],
            "lag_1": [None, 2.0],
            "lag_3": [None, 1.0],
            "lag_6": [None, 0.0],
            "lag_12": [None, 0.0],
        }
    )
    result = filter_complete_cases(grid)
    assert len(result) == 1
    assert result["lag_1"].isna().sum() == 0
