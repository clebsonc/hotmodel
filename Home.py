import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

import hotmodel.stats as stats
from hotmodel.data_loader import DatasetLoader

sns.set_theme()

st.title("Hotmodel Data")

st.markdown(
    """
    The provided dataset contains 50,000 anonymized samples in a CSV file.
    Each row represents a user who was either exposed or not exposed to the recommendation system.
    The columns signify the following:

    * id: index of the sample;
    * c1 to c6: categorical features of the user;
    * variant: the test variant that the user saw during the experiment;
    * n1: user engagement intensity (accumulated over the last 30 days before the test);
    * n2 to n12: numerical features of the user collected before the experiment started;
    * n13: user engagement intensity (calculated in the same way as n1) accumulated over the first
        7 days of the experiment;
    * n14: monetization metric (unknown unit. A higher value indicates a higher spending on
        purchases on the platform)
    """
)

st.header("First Look at the data")
st.write(
    """
    We are using `pandas.Dataframe` to analyse the dataset. The first thing that we should pay
    attention to is if the supposed columns does match wath the description suggests.
    Let\\`s load the data for the first time and glance at the raw data:
    """
)

dataloader = DatasetLoader(
    path="input/data.csv",
)

dataloader.load_data()
st.write(dataloader.data)

st.write(
    """
    The next thing we should pay attention to is if the data was properly loaded and parsed. This
    will make any matricial operation on the data work as expected.

    To check this, let\\`s first ensure that the data structure is exactly what it is supposed to
    be:
    """
)
dataloader.data.dtypes

st.header("Parsing data types")
st.write(
    """
    As it is possible to observe, most categorical features were loaded as `object`. This is really
    bad since matricial operations on objects are known to lazely evaluated, and if you have a large
    dataset (that fits in memory) the operations will take longer processing time. We should coerce
    its type to `string`.

    Lastly, let\\`s remove the column `id`, since it is just a mapping of earch row. It really
    doesn`t bring any real information regarding the user.

    Let`s verify the data type after coercion:
    """
)

categorical_features = ["c1", "c2", "c3", "c4", "c6", "variant"]
boolean_features = features = ["c5"]
# fmt: off
numerical_features = [
    "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14"
]
# fmt: on

dataloader.parse_column_types(
    categorical_columns=categorical_features,
    numerical_columns=numerical_features,
    boolean_columns=boolean_features,
)

dataloader.data.dtypes


st.header("Features with missing values")
st.write(
    """
    We should also pay some attention to column `c5`.
    The data description informs that this columns is a categorical column, however it was loaded as
    a `boolean` value. Let\\`s  verify some statistics regarding this feature for each test variant:
    """
)
st.write(stats.get_stats_by_variant(dataloader.data, col=["c5"]))

st.write(
    """
    As we can observe, the values for column C5 represent less than 1% of the total amount, and all
    of it\\`s values are for variant B."""
)

st.write(
    """
    Another interesting fact that is possible to observe here is that the column `n9`
    is mostly None. Let\\`s verify this information.
    """
)

miss = stats.get_percentage_missing_values(dataloader.data)

st.write(miss)

st.write(
    """
    The column `n9` has 95% of missing values. All remaining numerical features does not have
    missing values.

    We can also observe that the categorical features `c1`, `c2`, `c4`, and `c6` have more than 9%
    of missing values. Let`s check all of the remaining categorical features groups and count the
    instances for each group in order to better assess if there is a raw possibility of dropping
    these missing values:
    """
)

st.write("---")
cat_stats = stats.get_categorical_substats_by_variant_and_column(
    data=dataloader.data, col=["c1", "c2", "c3", "c4", "c6"]
)

c1, c2, c3 = st.columns(3)
c1.write("Aggregation for column C1")
c1.write(cat_stats["c1"]["primary"])
c1.write(cat_stats["c1"]["secondary"])

c2.write("Aggregation for column C2")
c2.write(cat_stats["c2"]["primary"])
c2.write(cat_stats["c2"]["secondary"])

c3.write("Aggregation for column C3")
c3.write(cat_stats["c3"]["primary"])
c3.write(cat_stats["c3"]["secondary"])

st.write("---")
c1, c2 = st.columns(2)
c1.write("Aggregation for column C4")
c1.write(cat_stats["c4"]["primary"])
c1.write(cat_stats["c4"]["secondary"])

c2.write("Aggregation for column C6")
c2.write(cat_stats["c6"]["primary"])
c2.write(cat_stats["c6"]["secondary"])
st.write("---")

st.write(
    """
    As it is possible to observe all categorical features for variant B have less than 14,720
    samples while the features for variant A have more than or near 30,000 samples. This shows us
    that the data set is quite unbalanced. There are lots of ways of solving this problem, bye
    either undersampling the data with most categories or oversampling the data with less 
    caretories.

    Since the purpose here is to show the code, let`s go with the dummiest possible way of
    resampling the dataset by just dropping all lines containing missing values for the categorical
    features for the variant with most samples, which is variant A.
    """
)


dataloader.data = dataloader.data.drop(["c5", "n9"], axis=1)

"""This is the dataset distribution for each column considering both variants before the undersample
of variant A."""
st.write(dataloader.data.groupby("variant").count())

temp = stats.undersample_col_with_na_with_categorical_group(
    data=dataloader.data, col="variant", group="A"
)

"""This is the dataset distribution for each column considering both variants after the undersample
of variant A."""
dataloader.data = temp

st.write(dataloader.data.groupby("variant").count())

st.write(
    """
    After the undersample, the only collum that have missing values is column `c2`.
    Let\\`s just ignore it and handle these missing values in the pipeline when building any model
    with this data.
    """
)


st.header("Checking Numerical Features")
st.write(
    """
    One of the many possible ways of verifying the distribution of numerical features is with a
    box and whiskers plot. The benefit of using this plot is the ease of identifying outliers
    and observe the percentiles of the distribution. Let\\`s visualize it for each individual
    feature:
    """

)

chosen = st.selectbox(
    label="select option: ",
    options=numerical_features,
    placeholder="Chose the numerical feature to analyze",
)
with st.container(border=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    sns.boxplot(ax=ax, data=dataloader.data[chosen], orient="v", notch=False)
    st.pyplot(fig)

    st.write(chosen * 3)


st.write("ii")
