import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


class DecodedModel:
    """Wraps a fitted XGB model + label encoder so .predict() returns original labels."""
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, X):
        encoded_predictions = self.model.predict(X) 
        return self.label_encoder.inverse_transform(encoded_predictions.astype(int))

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBOOST_TRAINER:

    def __init__(self):
        self.random_search = None
        self.best_model = None
        self.label_encoder = None

    def train_xgboost(
        self,
        X_train,
        y_train,
        cv_folds=10,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
    ):

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        self.label_encoder = LabelEncoder()
        y_train_encoded = (self.label_encoder.fit_transform(np.asarray(y_train)))

        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
        )

        self.random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=XGB_PARAM_GRID,
            n_iter=40,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=n_jobs,
            verbose=1,
            random_state=random_state,
            refit=True,
        )

        self.random_search.fit(X_train, y_train_encoded)
        self.best_model = self.random_search.best_estimator_

        return self.random_search, self.best_model

    def predict_xgboost(self, model, X_test):
        X_test = np.asarray(X_test)
        encoded_preds = model.predict(X_test)
        return self.label_encoder.inverse_transform(encoded_preds)

    def predict_proba_xgboost(self, model, X_test):
        X_test = np.asarray(X_test)
        return model.predict_proba(X_test)

    def get_best_params(self):
        return self.random_search.best_params_

    def get_best_cv_score(self):
        return self.random_search.best_score_