from itertools import product

import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub
from mlflow.data.numpy_dataset import NumpyDataset
from mlflow.data.pandas_dataset import PandasDataset
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from skore import CrossValidationReport, EstimatorReport
from skore._plugins.mlflow.reports import (
    Artifact,
    Dataset,
    Metric,
    Model,
    Tag,
    _dataset_from_Xy,
    _sample_input_example,
    iter_cv,
    iter_cv_metrics,
    iter_estimator,
    iter_estimator_metrics,
)
from skore._utils.skrub import find_fitted_estimators, is_skrub_learner

REPORT_FIXTURES = ["clf_report", "mclf_report", "reg_report", "mreg_report"]
CV_REPORT_FIXTURES = [
    "cv_clf_report",
    "cv_mclf_report",
    "cv_reg_report",
    "cv_mreg_report",
]


X_pandas = pd.DataFrame(
    {
        "cat": pd.Series(["a", "b", "c"], dtype="category"),
        "num": [1, 2, 3],
    }
)
X_numpy = np.identity(3)
y_numpy = np.arange(3)
y_pandas = pd.Series(["no cancer", "cancer", "no cancer"])
y_pandas_multi_targets = pd.DataFrame(
    {
        "label": pd.Series(["no cancer", "cancer", "no cancer"]),
        "confidence": [0.9, 0.6, 0.95],
    }
)
X_polars = pl.DataFrame({"cat": ["a", "b", "c"], "num": [1, 2, 3]})
y_polars = pl.Series("label", ["no cancer", "cancer", "no cancer"])
y_polars_multi = pl.DataFrame(
    {
        "label": ["no cancer", "cancer", "no cancer"],
        "confidence": [0.9, 0.6, 0.95],
    }
)


@pytest.fixture
def report(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("report", REPORT_FIXTURES, indirect=True)
def test_iter_estimator_metrics_smoke(report):
    assert all(
        isinstance(obj, Artifact | Metric) for obj in iter_estimator_metrics(report)
    )


@pytest.mark.parametrize("report", CV_REPORT_FIXTURES, indirect=True)
def test_iter_cv_metrics_smoke(report):
    assert all(isinstance(obj, Artifact | Metric) for obj in iter_cv_metrics(report))


@pytest.mark.parametrize("report", REPORT_FIXTURES, indirect=True)
def test_iter_estimator_smoke(report):
    items = list(iter_estimator(report))
    assert len({type(obj) for obj in items}) >= 3
    model = next(item for item in items if isinstance(item, Model))
    assert model.input_example is not None


@pytest.mark.parametrize("report", CV_REPORT_FIXTURES, indirect=True)
def test_iter_cv_smoke(report):
    items = list(iter_cv(report))
    assert len({type(obj) for obj in items}) >= 5
    model = next(item for item in items if isinstance(item, Model))
    assert model.input_example is not None


def test_sample_input_example_casts_category_to_object() -> None:
    sample = _sample_input_example(X_pandas, max_samples=2)

    assert sample.shape == (2, 2)
    assert not isinstance(sample["cat"].dtype, pd.CategoricalDtype)
    assert sample["cat"].tolist() == ["a", "b"]


@pytest.mark.parametrize(
    ("X", "y"),
    list(
        product(
            [X_pandas, X_numpy, X_polars],
            [y_pandas, y_numpy, y_pandas_multi_targets, y_polars, y_polars_multi],
        )
    ),
)
def test_dataset_from_Xy(X, y):
    x_type, y_type = type(X), type(y)
    logged = _dataset_from_Xy(X, y, context="training")

    # Conversion happens on a copy; the caller's frame or series is unchanged.
    assert type(X) is x_type
    assert type(y) is y_type
    assert logged.context == "training"
    dataset = logged.dataset
    assert isinstance(dataset, (PandasDataset, NumpyDataset))

    if isinstance(dataset, NumpyDataset):
        assert dataset.features.shape == X.shape
        if isinstance(dataset.targets, dict):
            assert list(dataset.targets) == list(y.columns)
            for key, value in dataset.targets.items():
                assert_array_equal(value, np.asarray(y[key]))
        else:
            assert_array_equal(np.asarray(dataset.targets), np.asarray(y))

    if isinstance(dataset, PandasDataset):
        target_col = getattr(y, "name", None) or "target"
        features = dataset.df.drop(columns=[target_col])
        if isinstance(X, pd.DataFrame):
            assert_array_equal(features, X)
        else:
            assert list(features.columns) == list(X.columns)
            assert_array_equal(features.to_numpy(), X.to_numpy())
        assert_array_equal(np.asarray(dataset.df[target_col]), np.asarray(y))
        assert list(dataset.df[target_col]) == list(np.asarray(y))


def test_dataset_from_Xy_aligns_polars_rows_to_pandas_index() -> None:
    X = X_pandas.copy()
    X.index = [5, 3, 1]
    logged = _dataset_from_Xy(X, y_polars, context="evaluation")

    assert logged.context == "evaluation"
    assert isinstance(y_polars, pl.Series)
    features = logged.dataset.df.drop(columns=["label"])
    assert list(logged.dataset.df.index) == [5, 3, 1]
    assert_array_equal(features, X)
    assert_array_equal(np.asarray(logged.dataset.df["label"]), np.asarray(y_polars))


def _weighted_regression_data_op():
    feat = np.linspace(0.0, 1.0, 40)
    frame = pd.DataFrame({"feat": feat})
    observed = pd.Series(feat * 2.0, name="target")
    features = skrub.var("df", frame)
    weight = skrub.var("weight", 2.0)
    X = (features[["feat"]] * weight).skb.mark_as_X()
    return X.skb.apply(LinearRegression(), y=skrub.y(observed))


@pytest.mark.parametrize("source", ["learner", "data_op"])
def test_skrub_iterators_refit_on_full_environment(source):
    """CV refits a clone on the full environment and does not touch the report."""
    data_op = _weighted_regression_data_op()
    if source == "data_op":
        cv_report = CrossValidationReport(data_op, splitter=KFold(n_splits=2))
        estimator_report = EstimatorReport(data_op)
    else:
        environment = data_op.skb.get_data()
        cv_report = CrossValidationReport(
            data_op.skb.make_learner(),
            data=environment,
            splitter=KFold(n_splits=2),
        )
        split = data_op.skb.train_test_split(environment, random_state=0)
        estimator_report = EstimatorReport(
            data_op.skb.make_learner(),
            train_data=split["train"],
            test_data=split["test"],
        )

    assert {"df", "weight"} <= set(cv_report.input_data)
    assert find_fitted_estimators(cv_report.learner_) == []

    cv_items = list(iter_cv(cv_report))
    assert any(isinstance(item, Metric) and item.name == "rmse" for item in cv_items)
    cv_model = next(item for item in cv_items if isinstance(item, Model))
    assert cv_model.input_example is None
    assert cv_model.model is not cv_report.learner_
    assert is_skrub_learner(cv_model.model)
    expected = clone(cv_report.learner_).fit(cv_report.input_data)
    assert_array_equal(
        cv_model.model.predict(cv_report.input_data),
        expected.predict(cv_report.input_data),
    )
    assert find_fitted_estimators(cv_report.learner_) == []
    assert not cv_report.learner_.__sklearn_is_fitted__()

    splits = [item for item in cv_items if isinstance(item, tuple)]
    assert [name for name, _ in splits] == ["split_0", "split_1"]
    for _, split_items in splits:
        nested = list(split_items)
        assert any(isinstance(item, Tag) and item.key == "split_id" for item in nested)
        assert any(isinstance(item, Metric) and item.name == "rmse" for item in nested)
        nested_model = next(item for item in nested if isinstance(item, Model))
        assert nested_model.input_example is None
        assert is_skrub_learner(nested_model.model)
        contexts = [item.context for item in nested if isinstance(item, Dataset)]
        assert contexts == ["training", "evaluation"]

    assert {"df", "weight"} <= set(estimator_report.test_data)
    fitted_ids = [id(est) for est in find_fitted_estimators(estimator_report.learner_)]
    before = estimator_report.estimator_.predict(estimator_report.test_data)
    estimator_items = list(iter_estimator(estimator_report))
    after = estimator_report.estimator_.predict(estimator_report.test_data)
    assert_array_equal(before, after)
    assert fitted_ids == [
        id(est) for est in find_fitted_estimators(estimator_report.learner_)
    ]
    assert any(
        isinstance(item, Metric) and item.name == "rmse" for item in estimator_items
    )
    estimator_model = next(item for item in estimator_items if isinstance(item, Model))
    assert estimator_model.input_example is None
    assert estimator_model.model is estimator_report.estimator_
    assert is_skrub_learner(estimator_model.model)
    contexts = [item.context for item in estimator_items if isinstance(item, Dataset)]
    assert contexts == ["training", "evaluation"]
