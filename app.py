import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from url_parser import extract_uci_url_features
from phishhooknet import PhishHookNet

app = Flask(__name__)
CORS(app)

# Load your trained model (ensure the model is available in the environment)
model = PhishHookNet(input_size=16)  # Adjust based on your actual model
model.load_state_dict(torch.load("phishing_url_model.pth"))
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    url = data["url"]
    print(url)
    features = extract_uci_url_features(url)
    print(features)
    feature_values = list(features.values())
    input_tensor = torch.tensor([feature_values], dtype=torch.float32)

    with torch.no_grad():
        probability = model(input_tensor).item()
        predicted_class = "Yes" if probability > 0.5 else "No"
        print(f"Predicted class: {predicted_class}, Probability: {probability}")

    return jsonify({"probability": probability, "is_phishing": predicted_class})


if __name__ == "__main__":
    app.run(debug=True)
