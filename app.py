from flask_cors import CORS
from flask import Flask, request, jsonify

from inference import infer

# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict if a URL is phishing or not.

    The input data should be a JSON object with a single key "url" that contains the URL to predict.

    Returns:
        dict: A dictionary containing the probability of the URL being a phishing URL and the predicted class.
    """
    # Get the URL from the request
    data = request.get_json(force=True)
    url = data["url"]

    # Make a prediction
    prediction = infer(url)

    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
