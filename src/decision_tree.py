from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from config import (
    DT_PARAM_GRID,
    CV_FOLDS,
    SCORING,
    RANDOM_STATE,
    N_JOBS,
)


class DecisionTreeTrainer:
    """Handles Decision Tree hyperparameter tuning."""

    def __init__(self):
        self.grid_search = None

    def train(self, X_train, y_train):
        """
        Perform Grid Search with cross-validation.

        Parameters
        ----------
        X_train : pandas.DataFrame
        y_train : pandas.Series

        Returns
        -------
        best_model
        best_params
        best_cv_score
        """

        model = DecisionTreeClassifier(
            random_state=RANDOM_STATE
        )

        self.grid_search = GridSearchCV(
            estimator=model,
            param_grid=DT_PARAM_GRID,
            scoring=SCORING,
            cv=CV_FOLDS,
            n_jobs=N_JOBS,
            refit=True,
            return_train_score=True,
        )

        self.grid_search.fit(X_train, y_train)

        return (
            self.grid_search.best_estimator_,
            self.grid_search.best_params_,
            self.grid_search.best_score_,
        )