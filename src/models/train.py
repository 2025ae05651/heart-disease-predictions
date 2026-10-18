"""
Model training script with MLflow experiment tracking.
Trains Logistic Regression and Random Forest classifiers.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_processed_data(data_dir: Path):
    """Load processed training and test data."""
    X_train = pd.read_csv(data_dir / "processed" / "X_train.csv")
    X_test = pd.read_csv(data_dir / "processed" / "X_test.csv")
    y_train = pd.read_csv(data_dir / "processed" / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir / "processed" / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression model with hyperparameter tuning."""
    print("Training Logistic Regression...")

    # Define hyperparameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'solver': ['liblinear']
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        LogisticRegression(random_state=random_state, max_iter=1000),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    return best_model, best_params


def train_random_forest(X_train, y_train, random_state=42):
    """Train Random Forest model with hyperparameter tuning."""
    print("Training Random Forest...")

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    return best_model, best_params


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    print(f"\n{model_name} Test Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, y_pred, y_pred_proba


def plot_confusion_matrix(y_test, y_pred, model_name, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    """Main training pipeline with MLflow tracking."""
    # Set up MLflow
    mlflow.set_experiment("Heart Disease Prediction")

    # Load data
    data_dir = Path("data")
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Create screenshots directory
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    # Train Logistic Regression
    with mlflow.start_run(run_name="Logistic Regression"):
        lr_model, lr_params = train_logistic_regression(X_train, y_train)

        # Log parameters
        mlflow.log_params(lr_params)

        # Evaluate
        lr_metrics, lr_pred, lr_pred_proba = evaluate_model(
            lr_model, X_test, y_test, "Logistic Regression"
        )

        # Log metrics
        mlflow.log_metrics(lr_metrics)

        # Log model
        mlflow.sklearn.log_model(lr_model, artifact_path="model")

        # Plot confusion matrix
        plot_confusion_matrix(
            y_test, lr_pred, "Logistic Regression",
            screenshots_dir / "05_lr_confusion_matrix.png"
        )

        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        joblib.dump(lr_model, model_dir / "logistic_regression.joblib")
        mlflow.log_artifact(model_dir / "logistic_regression.joblib")

    # Train Random Forest
    with mlflow.start_run(run_name="Random Forest"):
        rf_model, rf_params = train_random_forest(X_train, y_train)

        # Log parameters
        mlflow.log_params(rf_params)

        # Evaluate
        rf_metrics, rf_pred, rf_pred_proba = evaluate_model(
            rf_model, X_test, y_test, "Random Forest"
        )

        # Log metrics
        mlflow.log_metrics(rf_metrics)

        # Log model
        mlflow.sklearn.log_model(lr_model, artifact_path="model")

        # Plot confusion matrix
        plot_confusion_matrix(
            y_test, rf_pred, "Random Forest",
            screenshots_dir / "06_rf_confusion_matrix.png"
        )

        # Save model
        joblib.dump(rf_model, model_dir / "random_forest.joblib")
        mlflow.log_artifact(model_dir / "random_forest.joblib")

    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"{'Metric':<15} {'Logistic Regression':<20} {'Random Forest':<20}")
    print("-"*55)
    for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
        lr_val = lr_metrics[metric]
        rf_val = rf_metrics[metric]
        print(f"{metric:<15} {lr_val:<20.4f} {rf_val:<20.4f}")

    # Select best model (based on ROC-AUC)
    best_model_name = "Random Forest" if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else "Logistic Regression"
    best_model = rf_model if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else lr_model

    print(f"\nBest model: {best_model_name}")

    # Save best model as final model
    joblib.dump(best_model, model_dir / "best_model.joblib")
    print(f"Best model saved to {model_dir / 'best_model.joblib'}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
