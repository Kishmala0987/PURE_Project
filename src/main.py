from pathlib import Path
from feature_extractor import FeatureExtractor
from config import (
    FEATURE_SIZES,
    RANKING_METHODS,
    RANKING_DIR,
    MODEL_DIR,
    RESULT_DIR,
    DATA_DIR
)

from data_loader import DataLoader
from feature_selector import FeatureSelector
from xgboost_trainer import XGBOOST_TRAINER, DecodedModel
from evaluator import Evaluator
from model_manager import ModelManager
from results_manager import ResultsManager
MODEL_NAME = "xgboost"       

TARGETS = {
    "Task_ID": {
        "train": f"{DATA_DIR}/task_features.csv",
        "label": "Task_ID",
    },

    "Subject_ID": {
        "train": f"{DATA_DIR}/subject_features.csv",
        "label": "Subject_ID",
    },
}

TEST_RAW_DATA = f"{DATA_DIR}/test.csv"


def main():

    MAX_TOP_K = max(FEATURE_SIZES)
    trainer = XGBOOST_TRAINER()

    model_manager = ModelManager(
        base_dir=MODEL_DIR
    )

    for target, target_info in TARGETS.items():

        print("\n" + "=" * 70)
        print(f"Running {MODEL_NAME} experiments for {target}")
        print("=" * 70)

        X_train, y_train = DataLoader.load_dataset(
            target_info["train"]
        )
        for ranking_method in RANKING_METHODS:

            print(
                f"\n{'-' * 60}\n"
                f"Target: {target}\n"
                f"Ranking: {ranking_method}\n"
                f"{'-' * 60}"
            )
            ranking_path = (
                RANKING_DIR
                / f"{target}_{ranking_method}.csv"
            )

            ranking_df = DataLoader.load_ranking(
                ranking_path
            )

            _, max_features = (
                FeatureSelector.select_top_features(
                    X_train,
                    ranking_df,
                    MAX_TOP_K,
                )
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
                    f"\n{target} | "
                    f"{ranking_method} | "
                    f"Top {top_k}"
                )

                X_train_selected, selected_features = (
                    FeatureSelector.select_top_features(
                        X_train,
                        ranking_df,
                        top_k,
                    )
                )

                grid_search, best_model = (
                    trainer.train_xgboost(
                        X_train_selected,
                        y_train,
                    )
                )
                
                decoded_model = DecodedModel(best_model, trainer.label_encoder)

                X_test_selected = (
                    X_test_full[selected_features]
                )

                metrics = Evaluator.evaluate(
                    decoded_model,
                    X_test_selected,
                    y_test,
                )

                cm_path = (
                    RESULT_DIR
                    / MODEL_NAME
                    / target
                    / "confusion_matrices"
                    / f"{ranking_method}_Top{top_k}.csv"
                )

                ResultsManager.save_confusion_matrix(
                    confusion_matrix=metrics[
                        "confusion_matrix"
                    ],
                    labels=metrics["labels"],
                    save_path=cm_path,
                )
                model_path = model_manager.save_model(
                    model=best_model,
                    feature_names=selected_features,
                    best_params=trainer.get_best_params(),
                    target=target,
                    ranking_method=ranking_method,
                    n_features=top_k,
                    model_name= MODEL_NAME,
                    preprocessing=None,
                    cv_score=trainer.get_best_cv_score(),
                    label_encoder=trainer.label_encoder,  

                )
                result = {
                    "model": MODEL_NAME,
                    "target": target,
                    "ranking_method": ranking_method,
                    "top_features": top_k,
                    "cv_score": (
                        trainer.get_best_cv_score()
                    ),

                    "accuracy": metrics["accuracy"],
                    "f1_score": metrics["f1_score"],
                    **trainer.get_best_params(),
                }
                summary_path = (
                    RESULT_DIR
                    / MODEL_NAME
                    / target
                    / "summary.csv"
                )

                ResultsManager.save_summary(
                    result,
                    summary_path,
                )

    print("\n" + "=" * 70)
    print(f"{MODEL_NAME} training completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()

#7:16