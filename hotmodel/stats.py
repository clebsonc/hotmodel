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
