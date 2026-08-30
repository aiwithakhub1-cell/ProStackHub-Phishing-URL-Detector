import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "phishing_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)


# We deliberately use only features that can be
# reasonably related to the URL-analysis task.
FEATURE_COLUMNS = [
    "having_IPhaving_IP_Address ",
    "URLURL_Length ",
    "Shortining_Service ",
    "having_At_Symbol ",
    "double_slash_redirecting ",
    "Prefix_Suffix ",
    "having_Sub_Domain ",
    "SSLfinal_State ",
    "age_of_domain ",
    "HTTPS_token ",
]


def clean_columns(df):
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False
    )

    df = clean_columns(df)

    required = [
        column.strip()
        for column in FEATURE_COLUMNS
    ] + ["Result"]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # Keep only selected features + target
    df = df[
        required
    ].copy()

    # Force numeric values
    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert labels:
    # dataset uses -1 = legitimate
    # and 1 = phishing
    #
    # We map them to:
    # 0 = legitimate
    # 1 = phishing

    df["Result"] = (
        df["Result"]
        .apply(
            lambda value:
            1 if value == 1 else 0
        )
    )

    return df


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    ).tolist()

    return {
        "accuracy": round(
            accuracy,
            4
        ),
        "precision": round(
            precision,
            4
        ),
        "recall": round(
            recall,
            4
        ),
        "f1_score": round(
            f1,
            4
        ),
        "confusion_matrix": matrix
    }


def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    print("\nLoading dataset...")

    df = load_dataset()

    print(
        f"Rows used: {len(df)}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["Result"].value_counts().to_dict()
    )

    X = df[
        FEATURE_COLUMNS
    ]

    y = df["Result"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        "\nTraining Random Forest..."
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_metrics = evaluate_model(
        random_forest,
        X_test,
        y_test
    )

    print(
        "\nRandom Forest Metrics:"
    )

    print(
        json.dumps(
            rf_metrics,
            indent=4
        )
    )

    print(
        "\nTraining XGBoost..."
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_metrics = evaluate_model(
        xgb_model,
        X_test,
        y_test
    )

    print(
        "\nXGBoost Metrics:"
    )

    print(
        json.dumps(
            xgb_metrics,
            indent=4
        )
    )

    # Select best model by F1
    if (
        xgb_metrics["f1_score"]
        >
        rf_metrics["f1_score"]
    ):

        best_model_name = "XGBoost"
        best_model = xgb_model
        best_metrics = xgb_metrics

    else:

        best_model_name = "Random Forest"
        best_model = random_forest
        best_metrics = rf_metrics

    # Feature importance
    importance_values = (
        best_model.feature_importances_
    )

    feature_importance = {
        feature: round(
            float(importance),
            6
        )
        for feature, importance
        in zip(
            FEATURE_COLUMNS,
            importance_values
        )
    }

    feature_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # Save model + metadata
    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": feature_importance
    }

    model_path = os.path.join(
        MODEL_DIR,
        "phishing_detector.joblib"
    )

    joblib.dump(
        model_bundle,
        model_path
    )

    # Save metrics report
    report = {
        "dataset_rows": len(df),
        "features": FEATURE_COLUMNS,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "selected_model": best_model_name,
        "selected_model_metrics": best_metrics,
        "feature_importance": feature_importance
    }

    report_path = os.path.join(
        REPORT_DIR,
        "model_metrics.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        "\n================================"
    )

    print(
        f"Best Model: {best_model_name}"
    )

    print(
        f"Model saved: {model_path}"
    )

    print(
        f"Metrics saved: {report_path}"
    )

    print(
        "Training completed successfully."
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()