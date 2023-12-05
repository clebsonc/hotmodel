# Problem description

Based on the provided dataset, identify which is the best variant of the experiment for all users.
In other words, if you had to choose only one version of the experiment, which one would it be?

In addition of that, create a model that will recommend a test variant for a user,
in order to maximize their monetization, but without losing engagement.

To answer that, all details are given in the [Home](https://hotmodel.streamlit.app/) app page.


## Local instalation instructions

Works with python 3.11. Checkout the `pyproject.toml` file.

Install [pipx](https://github.com/pypa/pipx)
Install Poetry with Pipx: `pipx install poetry`
Run install command: `poetry install` or `pip install .` in an virtual environment.

## Running the app locally

Just run `./streamlit run Home.py` from the virtualen in which the instalation in the previous sections was made.
