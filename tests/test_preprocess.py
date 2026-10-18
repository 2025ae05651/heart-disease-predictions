"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import HeartDiseasePreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'age': [52, 53, 54, 55, 56],
        'sex': [1, 0, 1, 0, 1],
        'cp': [0, 1, 2, 0, 1],
        'trestbps': [125, 130, 135, 140, 145],
        'chol': [212, 250, 280, 300, 320],
        'fbs': [0, 1, 0, 1, 0],
        'restecg': [1, 0, 1, 2, 0],
        'thalach': [168, 170, 175, 180, 185],
        'exang': [0, 1, 0, 1, 0],
        'oldpeak': [1.0, 1.5, 2.0, 2.5, 3.0],
        'slope': [2, 1, 2, 0, 1],
        'ca': [2, 0, 1, 3, 0],
        'thal': [3, 2, 3, 1, 2],
        'target': [0, 1, 2, 0, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance."""
    return HeartDiseasePreprocessor()


def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization."""
    assert preprocessor.scaler is not None
    assert preprocessor.feature_columns is None


def test_clean_data(preprocessor, sample_data):
    """Test data cleaning."""
    cleaned = preprocessor.clean_data(sample_data)

    # Check that target is binary
    assert cleaned['target'].isin([0, 1]).all()

    # Check that numeric columns are numeric
    numeric_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(cleaned[col])


def test_get_feature_columns(preprocessor, sample_data):
    """Test getting feature columns."""
    cleaned = preprocessor.clean_data(sample_data)
    feature_cols = preprocessor.get_feature_columns(cleaned)

    assert 'target' not in feature_cols
    assert len(feature_cols) == 13


def test_split_data(preprocessor, sample_data):
    """Test data splitting."""
    cleaned = preprocessor.clean_data(sample_data)
    X_train, X_test, y_train, y_test = preprocessor.split_data(cleaned, test_size=0.4)

    # Check shapes
    assert X_train.shape[0] + X_test.shape[0] == cleaned.shape[0]
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]

    # Check that target is not in features
    assert 'target' not in X_train.columns
    assert 'target' not in X_test.columns


def test_scale_features(preprocessor, sample_data):
    """Test feature scaling."""
    cleaned = preprocessor.clean_data(sample_data)
    X_train, X_test, y_train, y_test = preprocessor.split_data(cleaned, test_size=0.4)

    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    # Check that scaling was applied
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape

    # Check that scaled data has mean close to 0 and std close to 1
    assert np.abs(X_train_scaled.mean()).max() < 1.0
    assert np.abs(X_train_scaled.std() - 1.0).max() < 0.5
