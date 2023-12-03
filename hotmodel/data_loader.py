from pathlib import Path

import pandas as pd


class DatasetLoader:
    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = path
        self._data = None

    @property
    def path(self):
        """The path property."""
        return self._path

    @path.setter
    def path(self, value: Path):
        value = Path(value)
        if value.exists():
            self._path = value
        else:
            raise FileExistsError(f"The given path does not exists: {value}")

    @property
    def data(self) -> pd.DataFrame:
        """The data property."""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame):
        self._data = value

    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.path)
        if data.empty:
            raise pd.errors.EmptyDataError
        self._data = data

    def parse_column_types(
        self,
        numerical_columns: list[str],
        categorical_columns: list[str],
        boolean_columns: list[str],
    ):
        if self._data is not None:
            self._data = self._data.drop("id", axis=1)
            self._data[numerical_columns] = self._data[numerical_columns].astype(float)
            self._data[categorical_columns] = self._data[categorical_columns].astype(
                pd.StringDtype()
            )
            self._data[boolean_columns] = self._data[boolean_columns].astype(bool)
