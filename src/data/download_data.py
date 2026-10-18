"""
Script to download the Heart Disease UCI Dataset.
"""

import os
import pandas as pd
from pathlib import Path


def download_heart_disease_data():
    """
    Download the Heart Disease UCI dataset from UCI ML Repository.
    """
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # UCI Heart Disease dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    print("Downloading Heart Disease UCI dataset...")
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        print(f"Dataset downloaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")

        # Save to CSV
        output_path = data_dir / "heart_disease_raw.csv"
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")

        # Display basic info
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        print("\nTarget distribution:")
        print(df['target'].value_counts())

        return df

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


if __name__ == "__main__":
    download_heart_disease_data()
