from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from predictor import predict_url


app = Flask(__name__)


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json(
        silent=True
    ) or {}

    url = str(
        data.get(
            "url",
            ""
        )
    ).strip()

    if not url:

        return jsonify({
            "error": "Please enter a URL."
        }), 400

    try:

        result = predict_url(
            url
        )

        return jsonify(
            result
        )

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