"""
main.py

Runs all Decision Tree experiments.
"""
from feature_extractor import FeatureExtractor

from pathlib import Path

from config import FEATURE_SIZES, RANKING_METHODS, RANKING_DIR, MODEL_DIR, RESULT_DIR
from data_loader import DataLoader
from feature_selector import FeatureSelector
from decision_tree import DecisionTreeTrainer
from evaluator import Evaluator
from model_manager import ModelManager
from results_manager import ResultsManager

def main():
    TARGETS = {
        "Task_ID": {
            "train": "../data/task_features.csv",
            "label": "Task_ID",
        },
    }

    TEST_RAW_DATA = "../data/test.csv"
    MAX_TOP_K = max(FEATURE_SIZES)

    trainer = DecisionTreeTrainer()

    for target, target_info in TARGETS.items():

        print(f"\nRunning experiments for {target}")

        X_train, y_train = DataLoader.load_dataset(
            target_info["train"]
        )

        for ranking_method in RANKING_METHODS:

            ranking_path = (
                RANKING_DIR
                / f"{target}_{ranking_method}.csv"
            )

            ranking_df = DataLoader.load_ranking(
                ranking_path
            )

            # --------------------------------------------------
            # Extract test features ONCE at the largest top_k
            # for this (target, ranking_method) pair. Smaller
            # top_k values are just column subsets of this,
            # since rankings are nested (top_50 ⊂ top_500).
            # --------------------------------------------------

            _, max_features = FeatureSelector.select_top_features(
                X_train,
                ranking_df,
                MAX_TOP_K,
            )

            X_test_full, y_test = (
                FeatureExtractor.extract_selected_features(
                    raw_test_file=TEST_RAW_DATA,
                    selected_features=max_features,
                    target=target,
                )
            )

            for top_k in FEATURE_SIZES:

                print(
                    f"{target} | "
                    f"{ranking_method} | "
                    f"Top {top_k}"
                )

                # --------------------------------------------------
                # Select training features
                # --------------------------------------------------

                X_train_selected, selected_features = (
                    FeatureSelector.select_top_features(
                        X_train,
                        ranking_df,
                        top_k,
                    )
                )

                # --------------------------------------------------
                # Train Decision Tree
                # --------------------------------------------------

                model, best_params, best_cv_score = trainer.train(
                    X_train_selected,
                    y_train,
                )

                # --------------------------------------------------
                # Slice pre-extracted test features for this top_k
                # (no re-extraction needed)
                # --------------------------------------------------

                X_test_selected = X_test_full[selected_features]

                # --------------------------------------------------
                # Evaluate
                # --------------------------------------------------

                metrics = Evaluator.evaluate(
                    model,
                    X_test_selected,
                    y_test,
                )
                cm_path = (
                    RESULT_DIR
                    / target
                    / "confusion_matrices"
                    / f"{ranking_method}_Top{top_k}.csv"
                )

                ResultsManager.save_confusion_matrix(
                    confusion_matrix=metrics["confusion_matrix"],
                    labels=metrics["labels"],
                    save_path=cm_path,
                )

                # --------------------------------------------------
                # Save model
                # --------------------------------------------------

                model_path = (
                    MODEL_DIR
                    / target
                    / f"{ranking_method}_Top{top_k}.joblib"
                )

                ModelManager.save_model(
                    model=model,
                    selected_features=selected_features,
                    hyperparameters=best_params,
                    preprocessing=None,
                    save_path=model_path,
                )


                result = {
                    "target": target,
                    "ranking_method": ranking_method,
                    "top_features": top_k,
                    "cv_score": best_cv_score,
                    "accuracy": metrics["accuracy"],
                    "f1_score": metrics["f1_score"],
                    **best_params,
                }

                summary_path = (
                    RESULT_DIR
                    / target
                    / "summary.csv"
                )

                ResultsManager.save_summary(
                    result,
                    summary_path,
                )


    print("\nTraining completed.")

if __name__ == "__main__":
    main()