"""
data_loader.py

Utilities for loading feature datasets and feature ranking files.
"""

from pathlib import Path

import pandas as pd


class DataLoader:
    """Loads datasets and ranking files."""

    @staticmethod
    def load_dataset(file_path: str | Path):
        """
        Load a feature dataset and split it into X and y.

        Parameters
        ----------
        file_path : str or Path
            Path to subject_features.csv or task_features.csv.

        Returns
        -------
        X : pandas.DataFrame
            Feature matrix.

        y : pandas.Series
            Target labels.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found:\n{file_path}")

        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(f"Dataset is empty:\n{file_path}")

        if df.shape[1] < 2:
            raise ValueError(
                "Dataset must contain at least one feature column and one label column."
            )

        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()

        return X, y

    @staticmethod
    def load_ranking(file_path: str | Path):
        """
        Load a feature ranking CSV.

        Parameters
        ----------
        file_path : str or Path

        Returns
        -------
        ranking_df : pandas.DataFrame
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Ranking file not found:\n{file_path}")

        ranking_df = pd.read_csv(file_path)

        if ranking_df.empty:
            raise ValueError(f"Ranking file is empty:\n{file_path}")

        required_columns = {"Rank", "feature"}

        missing = required_columns - set(ranking_df.columns)

        if missing:
            raise ValueError(
                f"Ranking file is missing required columns: {missing}"
            )

        return ranking_df