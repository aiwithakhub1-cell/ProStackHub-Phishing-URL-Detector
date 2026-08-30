import json
import os

from flask import Flask, render_template, request, jsonify

from predictor import predict_url


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/model")
def model():
    return render_template("model.html")


@app.route("/phishtank")
def phishtank():
    return render_template("phishtank.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    url = str(data.get("url", "")).strip()

    if not url:
        return jsonify({
            "error": "Please enter a URL."
        }), 400

    try:
        return jsonify(predict_url(url))

    except Exception as error:
        print(f"Prediction error: {error}")

        return jsonify({
            "error": str(error)
        }), 500


@app.route("/api/model-report")
def model_report():
    path = os.path.join(
        app.root_path,
        "reports",
        "model_metrics.json"
    )

    if not os.path.exists(path):
        return jsonify({
            "error": "Model metrics report not found."
        }), 404

    try:
        with open(path, "r", encoding="utf-8") as file:
            return jsonify(json.load(file))

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


@app.route("/api/phishtank-report")
def phishtank_report():
    path = os.path.join(
        app.root_path,
        "reports",
        "phishtank_validation.json"
    )

    if not os.path.exists(path):
        return jsonify({
            "error": "PhishTank validation report not found yet."
        }), 404

    try:
        with open(path, "r", encoding="utf-8") as file:
            return jsonify(json.load(file))

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5003,
        debug=False
    )