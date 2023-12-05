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
            f"""By observing the boxplot, it is possible to see that the feature `{chosen}`
            has the interquartile range of { dataloader.data[chosen].quantile(min_bound),
            dataloader.data[chosen].quantile(max_bound)} considering the quantiles of
            (lower, upper) bound of **{min_bound, max_bound}**.""",
        )
        st.markdown(
            """Quantile ranges here are computed with
            [pandas.Series.quantile](https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html)
            """
        )
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        sns.boxplot(ax=ax, data=dataloader.data[chosen], orient="v", notch=True)
        st.pyplot(fig)


def engagement_vs_revenue_multiplot(
    dataloader: DatasetLoader, group: str, engagement: str, revenue: str
):
    dist = (
        dataloader.data.reset_index()[["index", group, engagement, revenue]]
        .groupby(group)
        .agg({"index": "count"})
        .rename({"index": "count"}, axis=1)
    )
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.suptitle("Scatter and Barplot comparing User Engagement and Revenue for variant A and B\n")

    sns.barplot(data=dist, x=group, y="count", hue=group, ax=ax[0][0])
    ax[0][0].set_ylabel("Count of Users")
    ax[0][0].set_title("Bar plot: Distribution of users per variant")

    sns.scatterplot(data=dataloader.data, x=engagement, y=revenue, ax=ax[0][1], hue=group)
    ax[0][1].set_xlabel("User engagement")
    ax[0][1].set_ylabel("Revenue")
    ax[0][1].set_title("Scatter plot: User engagement by Revenue")

    # lets use the percentile interval (pi) for errorbar. The default is to use 95% -> [2.5, 97.5]
    sns.barplot(data=dataloader.data, x=group, y=revenue, hue=group, ax=ax[1][0], errorbar="pi")
    ax[1][0].set_title("\nBar plot: Percentile error Interval at 95% \nfor Revenue by Variant")
    ax[1][0].set_ylabel("Revenue")

    sns.barplot(data=dataloader.data, x=group, y=engagement, hue=group, ax=ax[1][1], errorbar="pi")
    ax[1][1].set_ylabel("User engagement")
    ax[1][1].set_title(
        "Bar plot: Percentile error Interval at 95% \nfor User Engagement by Variant"
    )

    fig.tight_layout()
    st.pyplot(fig)
