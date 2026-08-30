import json
import os

import requests
from dotenv import load_dotenv

from predictor import predict_url


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)

APP_KEY = os.getenv(
    "PHISHTANK_APP_KEY"
)

USER_AGENT = os.getenv(
    "PHISHTANK_USER_AGENT",
    "ProStackHub-Phishing-URL-Detector"
)

PHISHTANK_FEED = (
    "http://data.phishtank.com/data/online-valid.json"
)


def load_phishtank_urls(limit=10):

    headers = {
        "User-Agent": USER_AGENT
    }

    feed_url = PHISHTANK_FEED

    if APP_KEY:
        feed_url = (
            "http://data.phishtank.com/data/"
            f"{APP_KEY}/online-valid.json"
        )

    response = requests.get(
        feed_url,
        headers=headers,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    entries = data.get(
        "entries",
        []
    )

    urls = []

    for entry in entries:

        url = entry.get("url")

        if url:
            urls.append(url)

        if len(urls) >= limit:
            break

    return urls


def main():

    report_dir = os.path.join(
        BASE_DIR,
        "reports"
    )

    os.makedirs(
        report_dir,
        exist_ok=True
    )

    print(
        "Downloading recent verified phishing "
        "URLs from PhishTank..."
    )

    try:

        urls = load_phishtank_urls(
            limit=10
        )

    except Exception as error:

        print(
            f"PhishTank feed error: {error}"
        )

        return

    if not urls:

        print(
            "No URLs were returned by PhishTank."
        )

        return

    print(
        f"Loaded {len(urls)} URLs."
    )

    results = []

    for index, url in enumerate(
        urls,
        start=1
    ):

        print(
            f"\n[{index}/{len(urls)}] "
            f"Validating URL..."
        )

        try:

            prediction = predict_url(
                url
            )

            results.append({
                "url": url,
                "ml_prediction": prediction[
                    "prediction"
                ],
                "ml_probability": prediction[
                    "probability"
                ],
                "ml_confidence": prediction[
                    "confidence"
                ],
                "phishtank_reference": True
            })

            print(
                "ML prediction:",
                prediction["prediction"]
            )

        except Exception as error:

            results.append({
                "url": url,
                "error": str(error),
                "phishtank_reference": True
            })

    valid_predictions = [
        item
        for item in results
        if "ml_prediction" in item
    ]

    phishing_detected = sum(
        1
        for item in valid_predictions
        if item["ml_prediction"] == "Phishing"
    )

    legitimate_detected = sum(
        1
        for item in valid_predictions
        if item["ml_prediction"] == "Legitimate"
    )

    summary = {
        "source": "PhishTank online-valid feed",
        "sample_size": len(urls),
        "successful_predictions": len(
            valid_predictions
        ),
        "predicted_phishing": phishing_detected,
        "predicted_legitimate": legitimate_detected
    }

    report = {
        "summary": summary,
        "results": results
    }

    output_file = os.path.join(
        report_dir,
        "phishtank_validation.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        "\n===================================="
    )

    print(
        "PhishTank validation completed."
    )

    print(
        f"Sample size: {len(urls)}"
    )

    print(
        f"Predicted phishing: "
        f"{phishing_detected}"
    )

    print(
        f"Predicted legitimate: "
        f"{legitimate_detected}"
    )

    print(
        f"Report saved to:\n{output_file}"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()