import pandas as pd

class FeatureSelector:
    """Selects top-ranked features."""

    @staticmethod
    def select_top_features(
        X: pd.DataFrame,
        ranking_df: pd.DataFrame,
        top_k: int,
    ):
        """
        Select the top-k ranked features.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        ranking_df : pandas.DataFrame
            Ranking dataframe.

        top_k : int
            Number of top features to keep.

        Returns
        -------
        X_selected : pandas.DataFrame

        selected_features : list[str]
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if top_k > len(ranking_df):
            raise ValueError(
                f"Requested {top_k} features but ranking contains only {len(ranking_df)}."
            )

        selected_features = ranking_df["feature"].head(top_k).tolist()

        missing = set(selected_features) - set(X.columns)

        if missing:
            raise ValueError(
                f"{len(missing)} ranked features are not present in the dataset."
            )

        X_selected = X[selected_features].copy()

        return X_selected, selected_features