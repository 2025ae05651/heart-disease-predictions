"""
Exploratory Data Analysis (EDA) for Heart Disease Dataset.
This script generates professional visualizations and saves them to the screenshots folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_data(filepath: str) -> pd.DataFrame:
    """Load the cleaned dataset."""
    df = pd.read_csv(filepath)
    return df


def plot_target_distribution(df: pd.DataFrame, save_path: str):
    """Plot target class distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))

    target_counts = df['target'].value_counts()
    colors = ['#3498db', '#e74c3c']
    labels = ['No Heart Disease', 'Heart Disease']

    ax.bar(labels, target_counts.values, color=colors, alpha=0.7)
    ax.set_xlabel('Target Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title('Target Class Distribution', fontsize=16, fontweight='bold')

    # Add count labels on bars
    for i, count in enumerate(target_counts.values):
        ax.text(i, count + 2, str(count), ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Target distribution plot saved to {save_path}")


def plot_feature_distributions(df: pd.DataFrame, save_path: str):
    """Plot histograms for all numerical features."""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('target')

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(numerical_features):
        if idx < len(axes):
            axes[idx].hist(df[feature].dropna(), bins=20, color='#3498db', alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel(feature, fontsize=10, fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'Distribution of {feature}', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature distributions plot saved to {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)

    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to {save_path}")


def plot_feature_by_target(df: pd.DataFrame, save_path: str):
    """Plot key features by target class."""
    key_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(key_features):
        if idx < len(axes):
            for target_val in [0, 1]:
                data = df[df['target'] == target_val][feature]
                label = 'No Disease' if target_val == 0 else 'Disease'
                color = '#3498db' if target_val == 0 else '#e74c3c'
                axes[idx].hist(data, bins=20, alpha=0.5, label=label, color=color, edgecolor='black')

            axes[idx].set_xlabel(feature, fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].set_title(f'{feature} by Target', fontsize=12, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature by target plot saved to {save_path}")


def generate_summary_statistics(df: pd.DataFrame, save_path: str):
    """Generate and save summary statistics."""
    summary = df.describe()
    summary.to_csv(save_path)
    print(f"Summary statistics saved to {save_path}")


def main():
    """Main EDA pipeline."""
    # Create screenshots directory
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    # Load cleaned data
    data_dir = Path("data")
    cleaned_data_path = data_dir / "heart_disease_cleaned.csv"

    if not cleaned_data_path.exists():
        print(f"Error: {cleaned_data_path} not found. Please run preprocess.py first.")
        return

    print("Loading cleaned data for EDA...")
    df = load_data(cleaned_data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nSummary statistics:")
    print(df.describe())

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_target_distribution(
        df,
        screenshots_dir / "01_target_distribution.png"
    )

    plot_feature_distributions(
        df,
        screenshots_dir / "02_feature_distributions.png"
    )

    plot_correlation_heatmap(
        df,
        screenshots_dir / "03_correlation_heatmap.png"
    )

    plot_feature_by_target(
        df,
        screenshots_dir / "04_features_by_target.png"
    )

    # Generate summary statistics
    generate_summary_statistics(
        df,
        screenshots_dir / "summary_statistics.csv"
    )

    print("\nEDA complete! All visualizations saved to screenshots/ directory.")


if __name__ == "__main__":
    main()
