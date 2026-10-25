

import mlflow

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.data_preprocessing import create_preprocessor, split_data, create_x_y
from src.load_data import load_data
from src.data_cleaning import clean_data
from src.metrics import compute_metrics


def run_model(name, classifier, X_train, y_train, X_test, y_test):
    preprocessor = create_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "model": name,
            **classifier.get_params(),
        })

        model.fit(X_train, y_train)

        model_pred = model.predict(X_test)
        model_scores = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test,model_pred,model_scores,plot=False,label=name)

        mlflow.log_metrics(metrics)

    return metrics


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("telco_customer_churn_prediction")

    df = clean_data(load_data())

    X, y = create_x_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            random_state=42,
            probability=True,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            objective="binary:logistic",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        ),
    }

    results = {}

    for name, classifier in models.items():
        results[name] = run_model(name,classifier,X_train,y_train,X_test,y_test)

    best_model_test=max(results,key=lambda name:results[name]["PR-AUC"])
    print(best_model_test)

    for name, metrics in results.items():
        print(name, metrics)


if __name__ == "__main__":
    main()