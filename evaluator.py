from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)


class Evaluator:
    """Evaluates trained classification models."""

    @staticmethod
    def evaluate(model, X_test, y_test):
        """
        Evaluate a trained model.

        Parameters
        ----------
        model : sklearn estimator

        X_test : pandas.DataFrame

        y_test : pandas.Series

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
        """

        predictions = model.predict(X_test)
        labels = sorted(y_test.unique())
        results = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(
                y_test,
                predictions,
                average="macro",
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels = labels
            ),
            "labels": labels,
        }

        return results