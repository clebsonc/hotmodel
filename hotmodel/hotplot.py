import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from hotmodel.data_loader import DatasetLoader

sns.set_theme()


def numerical_feature_container_boxplot(
    dataloader: DatasetLoader, key: int, min_bound: float = 0.05, max_bound: float = 0.95
):
    with st.container(border=True):
        chosen = st.selectbox(
            label="Select numerical feature:",
            options=dataloader.numerical_feature_names,
            placeholder="Chose the numerical feature to analyze",
            key=key,
        )
        st.markdown(
            f"""By observing the boxplot, it is possible to observe that the feature `{chosen}`
            has the interquantile range of { dataloader.data[chosen].quantile(min_bound),
            dataloader.data[chosen].quantile(max_bound)} considering the quantiles of
            (lower, upper) bound of **{min_bound, max_bound}**.""",
        )
        st.markdown(
            """Quantile ranges here compute with
            [pandas.Series.quantile](https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html)
            """
        )
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        sns.boxplot(ax=ax, data=dataloader.data[chosen], orient="v", notch=True)
        st.pyplot(fig)
