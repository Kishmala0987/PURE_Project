"""
results_manager.py

Utilities for saving experiment results.
"""

from pathlib import Path

import pandas as pd


class ResultsManager:
    """Handles saving experiment results."""

    @staticmethod
    def save_summary(result, save_path):
        """
        Append one experiment to the summary CSV.

        Parameters
        ----------
        result : dict

        save_path : str or Path
        """

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([result])

        if save_path.exists():
            df.to_csv(
                save_path,
                mode="a",
                header=False,
                index=False,
            )
        else:
            df.to_csv(
                save_path,
                index=False,
            )

    @staticmethod
    def save_confusion_matrix(
        confusion_matrix,
        labels,
        save_path,
    ):
        """
        Save a confusion matrix as a CSV.

        Parameters
        ----------
        confusion_matrix : ndarray

        labels : list

        save_path : str or Path
        """

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm_df = pd.DataFrame(
            confusion_matrix,
            index=labels,
            columns=labels,
        )

        cm_df.to_csv(save_path)