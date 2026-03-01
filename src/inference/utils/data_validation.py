"""
src/utils/data_validation.py
-----------------------------
Reusable functions to validate DataFrames before processing or modeling.
"""

# -----
# Libraries
import pandas as pd


# -----
# Verify required columns
def verify_required_columns(
    dataframe: pd.DataFrame, required_columns: list[str], dataset_name: str = "dataset"
) -> None:
    """
    Raise a ValueError if any required column is missing from the DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to validate.
    required_columns : list[str]
        List of column names that must exist.
    dataset_name : str
        Descriptive name of the dataset (used in the error message).

    Raises
    ------
    ValueError
        If any of the required columns is not in the DataFrame.
    """
    missing_columns = [c for c in required_columns if c not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"[{dataset_name}] Missing columns: {missing_columns}")


# -----
# Verify no nulls
def verify_no_nulls(
    dataframe: pd.DataFrame, columns: list[str], dataset_name: str = "dataset"
) -> None:
    """
    Raise a ValueError if any of the specified columns contains null values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to validate.
    columns : list[str]
        Columns to check for null values.
    dataset_name : str
        Descriptive name of the dataset (used in the error message).

    Raises
    ------
    ValueError
        If any column contains NaN values.
    """
    columns_with_nulls = [c for c in columns if dataframe[c].isna().any()]
    if columns_with_nulls:
        raise ValueError(
            f"[{dataset_name}] Columns with null values: {columns_with_nulls}"
        )


# -----
# Has enough rows
def has_enough_rows(dataframe: pd.DataFrame, minimum: int = 1) -> bool:
    """
    Return True if the DataFrame has at least `minimum` rows.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to check.
    minimum : int
        Minimum number of expected rows.

    Returns
    -------
    bool
        True if len(dataframe) >= minimum, False otherwise.
    """
    return len(dataframe) >= minimum
