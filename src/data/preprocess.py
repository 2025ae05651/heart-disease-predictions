"""
Data preprocessing module for Heart Disease dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class HeartDiseasePreprocessor:
    """Preprocessor for Heart Disease dataset."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the raw dataset."""
        df = pd.read_csv(filepath)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling missing values."""
        # Replace missing values (marked as ? in original dataset)
        df = df.replace('?', np.nan)

        # Drop rows with missing values
        df = df.dropna()

        # Convert columns to numeric
        numeric_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert target to binary (0 = no disease, 1 = disease)
        # Original dataset has values 0-4, we'll map 0 to 0 and 1-4 to 1
        df['target'] = (df['target'] > 0).astype(int)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get feature column names (excluding target)."""
        feature_cols = [col for col in df.columns if col != 'target']
        self.feature_columns = feature_cols
        return feature_cols

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2,
                   random_state: int = 42) -> tuple:
        """Split data into train and test sets."""
        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def save_preprocessor(self, filepath: str):
        """Save the preprocessor object."""
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load_preprocessor(filepath: str):
        """Load a saved preprocessor object."""
        return joblib.load(filepath)


def main():
    """Main preprocessing pipeline."""
    # Initialize preprocessor
    preprocessor = HeartDiseasePreprocessor()

    # Load data
    data_dir = Path("data")
    raw_data_path = data_dir / "heart_disease_raw.csv"

    if not raw_data_path.exists():
        print(f"Error: {raw_data_path} not found. Please run download_data.py first.")
        return

    print("Loading raw data...")
    df = preprocessor.load_data(raw_data_path)

    # Clean data
    print("Cleaning data...")
    df_clean = preprocessor.clean_data(df)
    print(f"Cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    # Store feature names and their order
    preprocessor.get_feature_columns(df_clean)
    # Save cleaned data
    cleaned_data_path = data_dir / "heart_disease_cleaned.csv"
    df_clean.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved to {cleaned_data_path}")

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_clean)
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Scale features
    print("Scaling features...")
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    # Save processed data
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=preprocessor.feature_columns).to_csv(
        processed_dir / "X_train.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=preprocessor.feature_columns).to_csv(
        processed_dir / "X_test.csv", index=False
    )
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    print(f"Processed data saved to {processed_dir}")

    # Save scaler and feature columns separately
    joblib.dump(preprocessor.scaler, data_dir / "scaler.joblib")
    joblib.dump(preprocessor.feature_columns, data_dir / "feature_columns.joblib")
    print(f"Scaler saved to {data_dir / 'scaler.joblib'}")
    print(f"Feature columns saved to {data_dir / 'feature_columns.joblib'}")

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
