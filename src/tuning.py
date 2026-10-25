import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_preprocessing import create_preprocessor, create_x_y, split_data
from src.load_data import load_data
from src.data_cleaning import clean_data
from src.metrics import compute_metrics


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("telco_customer_churn_prediction")

    # Load the same cleaned data used in baseline training.
    df = clean_data(load_data())

    X, y = create_x_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Fresh pipeline for GridSearchCV to fit and tune.
    preprocessor = create_preprocessor(X_train)

    lr_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        )),
    ])

    cv = StratifiedKFold( n_splits=5,shuffle=True,random_state=42)

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": [None, "balanced"]
    }

    with mlflow.start_run(
        run_name="hyperparameter-tuning-logistic-regression"
    ):
        mlflow.log_params({
            "search_method": "GridSearchCV",
            "estimator": "LogisticRegression",
            "param_grid": str(param_grid),
            "scoring": "average_precision",
            "cv_folds": 5,
            "n_jobs": 1,
            "refit": True,
            "random_state": 42
        })

        search = GridSearchCV(
            estimator=lr_model,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            refit=True
        )

        # Tune using training data only.
        search.fit(X_train, y_train)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_pr_auc", search.best_score_)

        print("Best parameters:", search.best_params_)
        print("Best CV PR-AUC:", search.best_score_)

        # Best pipeline is refitted on all training data.
        best_lr = search.best_estimator_

        # final test evaluation.
        best_pred = best_lr.predict(X_test)
        best_scores = best_lr.predict_proba(X_test)[:, 1]

        test_metrics = compute_metrics(y_test,best_pred,best_scores,plot=True,label="Tuned Logistic Regression",)

        mlflow.log_metrics(test_metrics)

        # Save the whole pipeline: preprocessing plus classifier.
        mlflow.sklearn.log_model(
            best_lr,
            name="best_lr_model",
            registered_model_name="churn_prediction_model"
        )


if __name__ == "__main__":
    main()