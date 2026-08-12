from pathlib import Path
import pickle


class ModelManager:
    """Handles saving and loading trained models."""

    def __init__(self, base_dir="models"):
        self.base_dir = Path(base_dir)

    def get_model_path(
        self,
        model_name,
        target,
        ranking_method,
        n_features,
    ):
        """
        Create and return the model path.

        Example:
        models/xgboost/Task_ID/ANOVA_Only/Top_50_model.pkl
        """

        model_dir = (
            self.base_dir
            / model_name
            / target
            / ranking_method
            /f"{ranking_method}_Top_{n_features}"
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return model_dir / "model.pkl"

    def save_model(
        self,
        model,
        feature_names,
        best_params,
        target,
        ranking_method,
        n_features,
        model_name="xgboost",
        preprocessing=None,
        cv_score=None,
        label_encoder=None,

    ):

        model_path = self.get_model_path(
            model_name=model_name,
            target=target,
            ranking_method=ranking_method,
            n_features=n_features,
        )

        artifact = {
            "model": model,
            "feature_names": list(feature_names),
            "best_params": best_params,
            "target": target,
            "ranking_method": ranking_method,
            "n_features": n_features,
            "preprocessing": preprocessing,
            "cv_score": cv_score,
            "label_encoder": label_encoder,   

        }

        with open(model_path, "wb") as f:
            pickle.dump(
                artifact,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return model_path

    @staticmethod
    def load_model(model_path):
        model_path = Path(model_path)

        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        return artifact