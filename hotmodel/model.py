from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


class HotModelClassifier:
    def __init__(self, data: pd.DataFrame, features: list[str], hyperparameters: dict[str, Any]):
        self.data = data
        self.features = features
        self.hyperparameters = hyperparameters

    def build_ordinal_enconder(
        self,
        ordinal_features: list[str],
        unknown_value: int = -1,
        encoded_missing_value: int = -1,
        min_frequency: int = 100,
    ):
        encoder = OrdinalEncoder(
            encoded_missing_value=encoded_missing_value,
            unknown_value=unknown_value,
            min_frequency=min_frequency,
            handle_unknown="use_encoded_value",
        )
        return "ordinal_encoder", encoder, ordinal_features

    def build_one_hot_encoder(
        self,
        data: pd.DataFrame,
        one_hot_features: list[str],
    ):
        print(len(data), len(one_hot_features))
        raise NotImplementedError("The one hot encoder transformer is not ready yet.")

    def pipeline_builder(self, ordinal_features: list[str], one_hot_features: list[str]):
        transformers = []
        transformers.append(self.build_ordinal_enconder(ordinal_features=ordinal_features))

        # Todo: implement the logic for hot encoder transformer.
        # transformers.append(build_one_hot_encoder(ordinal_features=ordinal_features))

        column_transformer = ColumnTransformer(transformers)
        preprocessors = [("column_transformer", column_transformer)]
        self.pipeline = Pipeline(preprocessors)
        df_transformed = self.pipeline.fit_transform(self.data)

        new_col_names = [x.split("__")[1] for x in self.pipeline.get_feature_names_out()]
        df_transformed = pd.DataFrame(df_transformed, columns=new_col_names, index=self.data.index)
        data = self.data.drop(new_col_names, axis=1)
        data = data.join(df_transformed)
        return data

    def train(self, data: pd.DataFrame, target: str):
        self.label_encoder = LabelEncoder()
        data[target] = self.label_encoder.fit_transform(data.loc[:, "variant"])

        model = RandomForestClassifier(**self.hyperparameters)
        model.fit(X=data.loc[:, self.features], y=data[target])
        self.model = model

    def predict(self, payload: pd.DataFrame):
        payload_transformed = self.pipeline.transform(payload)
        new_col_names = [x.split("__")[1] for x in self.pipeline.get_feature_names_out()]
        payload_transformed = pd.DataFrame(
            payload_transformed, columns=new_col_names, index=payload.index
        )
        payload = payload.drop(new_col_names, axis=1)
        payload = payload.join(payload_transformed)
        result = self.model.predict(payload.loc[:, self.features])
        return self.label_encoder.inverse_transform(result)

    def compute_evaluation_metric(self):
        raise NotImplementedError("Work in progress...")

    def cross_validate(self, data: pd.DataFrame, fold_size: int):
        raise NotImplementedError("Work in progress...")
