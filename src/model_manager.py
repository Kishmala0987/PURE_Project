"""
model_manager.py

Utilities for saving and loading trained models.
"""

from pathlib import Path
import joblib


class ModelManager:
    """Save and load trained models."""

    @staticmethod
    def save_model(
        model,
        selected_features,
        hyperparameters,
        save_path,
        preprocessing=None,
    ):
        """
        Save a trained model and its metadata.

        Parameters
        ----------
        model : sklearn estimator

        feature_names : list[str]

        hyperparameters : dict

        save_path : str or Path

        preprocessing : optional
        """

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_package = {
            "model": model,
            "selected_features": selected_features,
            "hyperparameters": hyperparameters,
            "preprocessing": preprocessing,
        }

        joblib.dump(model_package, save_path)

    @staticmethod
    def load_model(model_path):
        """
        Load a previously saved model package.
        """

        return joblib.load(model_path)