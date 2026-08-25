import json
import os

import joblib
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "phishing_detector.joblib"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "model_metrics.json"
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. "
            "Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def load_report():
    if not os.path.exists(REPORT_PATH):
        return {}

    with open(
        REPORT_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def extract_url_features(url):
    """
    Convert a user-entered URL into the same
    10-feature format used by the trained model.

    This is a lightweight structural extractor.
    The dataset itself uses -1 / 0 / 1 values,
    so we map observable URL characteristics
    into that same range.
    """

    text = str(url).strip()

    lowered = text.lower()

    # 1. IP address
    hostname_part = lowered

    if "://" in hostname_part:
        hostname_part = hostname_part.split(
            "://",
            1
        )[1]

    hostname_part = hostname_part.split(
        "/",
        1
    )[0]

    hostname_part = hostname_part.split(
        ":",
        1
    )[0]

    ip_like = 0

    parts = hostname_part.split(".")

    if len(parts) == 4:
        try:
            if all(
                0 <= int(part) <= 255
                for part in parts
            ):
                ip_like = 1
        except ValueError:
            ip_like = 0

    # 2. URL length
    if len(text) < 54:
        url_length = -1
    elif len(text) <= 75:
        url_length = 0
    else:
        url_length = 1

    # 3. URL shortening service
    shorteners = [
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly"
    ]

    shortening_service = int(
        any(
            shortener in lowered
            for shortener in shorteners
        )
    )

    # 4. @ symbol
    at_symbol = int(
        "@" in text
    )

    # 5. Double slash redirect
    after_scheme = text

    if "://" in after_scheme:
        after_scheme = after_scheme.split(
            "://",
            1
        )[1]

    double_slash = int(
        "//" in after_scheme
    )

    # 6. Prefix / suffix
    prefix_suffix = int(
        "-" in hostname_part
    )

    # 7. Subdomain count
    labels = [
        label
        for label in hostname_part.split(".")
        if label
    ]

    if len(labels) <= 2:
        subdomain = -1
    elif len(labels) == 3:
        subdomain = 0
    else:
        subdomain = 1

    # 8. HTTPS
    https_state = (
        1
        if lowered.startswith("https://")
        else -1
    )

    # 9. Domain age
    # We do not perform a live WHOIS lookup here.
    # The dataset uses categorical age values.
    # Unknown live age is represented neutrally.
    domain_age = 0

    # 10. HTTPS token
    https_token = int(
        "https" in hostname_part
        or "@https" in lowered
    )

    return {
        "having_IPhaving_IP_Address": ip_like,
        "URLURL_Length": url_length,
        "Shortining_Service": (
            1
            if shortening_service
            else -1
        ),
        "having_At_Symbol": (
            1
            if at_symbol
            else -1
        ),
        "double_slash_redirecting": (
            1
            if double_slash
            else -1
        ),
        "Prefix_Suffix": (
            1
            if prefix_suffix
            else -1
        ),
        "having_Sub_Domain": subdomain,
        "SSLfinal_State": https_state,
        "age_of_domain": domain_age,
        "HTTPS_token": (
            1
            if https_token
            else -1
        )
    }


def predict_url(url):

    bundle = load_model()

    model = bundle["model"]

    feature_columns = bundle[
        "feature_columns"
    ]

    features = extract_url_features(
        url
    )

    feature_values = {
        column: features.get(
            column,
            0
        )
        for column in feature_columns
    }

    X = pd.DataFrame(
        [feature_values],
        columns=feature_columns
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    prediction = int(
        model.predict(X)[0]
    )

    result = (
        "Phishing"
        if prediction == 1
        else "Legitimate"
    )

    confidence = (
        probability
        if prediction == 1
        else 1 - probability
    )

    report = load_report()

    importance = report.get(
        "feature_importance",
        {}
    )

    ranked_factors = []

    for feature, value in feature_values.items():

        ranked_factors.append({
            "feature": feature,
            "value": value,
            "importance": round(
                float(
                    importance.get(
                        feature,
                        0
                    )
                ),
                4
            )
        })

    ranked_factors.sort(
        key=lambda item:
        item["importance"],
        reverse=True
    )

    return {
        "url": url,
        "prediction": result,
        "probability": round(
            probability * 100,
            2
        ),
        "confidence": round(
            confidence * 100,
            2
        ),
        "threshold": 50.0,
        "features": feature_values,
        "top_factors": ranked_factors[:5]
    }
