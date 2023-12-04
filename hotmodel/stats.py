from __future__ import annotations

import pandas as pd
from pandas.errors import InvalidColumnName


def get_stats_by_variant(data: pd.DataFrame, col: list[str]) -> pd.DataFrame:
    if isinstance(col, list):
        col = ["variant"] + col
        data = (
            data[col]
            .reset_index()
            .groupby(col)
            .agg({"index": "count"})
            .rename(columns={"index": "count"})
        )
        data["percentage"] = round(data["count"] / data["count"].sum() * 100, 2)
        return data
    else:
        raise InvalidColumnName("Give a list of `valid` column names")


def get_categorical_substats_by_variant_and_column(
    data: pd.DataFrame, col: list[str]
) -> dict[str, pd.DataFrame]:
    result = {}
    for c in col:
        a = get_stats_by_variant(data, col=[c])
        b = a.groupby(["variant"]).agg({"count": "sum"})

        result[c] = {"primary": a, "secondary": b}
    return result


def get_percentage_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    temp = (
        data.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "feature_name", 0: "missing_values"})
    )
    temp["percentage"] = temp["missing_values"] / data.shape[0] * 100
    return temp


def undersample_col_with_na_with_categorical_group(
    data: pd.DataFrame, col: str, group: str
) -> pd.DataFrame:
    temp_a = data[data[col] == group].dropna(axis=0, how="any")
    temp_b = data[data[col] != group]
    temp = data.loc[list(temp_a.index) + list(temp_b.index), :]
    return temp


def multi_col_clip(
    data: pd.DataFrame,
    cols: list[str],
    quantile_lower_bound: float = 0.01,
    quantile_upper_bound: float = 0.99,
):
    data = data.copy(deep=True)
    for c in cols:
        min_bound = data[c].quantile(quantile_lower_bound)
        max_bound = data[c].quantile(quantile_upper_bound)

        # This is a limit that allow us to better visualize samples.
        # The table makes any number greather than 2**53 as `inf`.
        # if making matricial operations on this number, everything will be `inf`
        max_bound = max_bound if max_bound < 2**53 else 2**52

        data[c] = data[c].clip(min_bound, max_bound)
    return data
