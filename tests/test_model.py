"""
Unit tests for model training module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_train_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    X_train = np.random.randn(100, 13)
    y_train = np.random.randint(0, 2, 100)
    return X_train, y_train


@pytest.fixture
def sample_test_data():
    """Create sample test data for testing."""
    np.random.seed(42)
    X_test = np.random.randn(20, 13)
    y_test = np.random.randint(0, 2, 20)
    return X_test, y_test


def test_logistic_regression_training(sample_train_data):
    """Test Logistic Regression training."""
    X_train, y_train = sample_train_data

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Check that model is trained
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')

    # Check prediction
    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train)
    assert np.issubdtype(predictions.dtype, np.integer)


def test_random_forest_training(sample_train_data):
    """Test Random Forest training."""
    X_train, y_train = sample_train_data

    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X_train, y_train)

    # Check that model is trained
    assert hasattr(model, 'feature_importances_')
    assert hasattr(model, 'n_estimators')

    # Check prediction
    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train)
    assert np.issubdtype(predictions.dtype, np.integer)


def test_model_prediction(sample_train_data, sample_test_data):
    """Test model prediction on test data."""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data

    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Check predictions
    assert len(y_pred) == len(y_test)
    assert y_pred_proba.shape == (len(y_test), 2)

    # Check that probabilities sum to 1
    assert np.allclose(y_pred_proba.sum(axis=1), 1.0)


def test_model_metrics(sample_train_data, sample_test_data):
    """Test model evaluation metrics."""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data

    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Check that metrics are in valid range
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= roc_auc <= 1
