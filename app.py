import torch
from flask import Flask, request, jsonify

from url_parser import extract_url_features
from phishhooknet import PhishHookNet

app = Flask(__name__)

# Load your trained model (ensure the model is available in the environment)
model = PhishHookNet(input_size=23)  # Adjust based on your actual model
model.load_state_dict(torch.load('phishing_url_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    url = data['url']
    features = extract_url_features(url)
    feature_values = list(features.values())
    input_tensor = torch.tensor([feature_values], dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_prob = torch.sigmoid(output).item()
        predicted_class = int(predicted_prob > 0.5)

    return jsonify({'probability': predicted_prob, 'is_phishing': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)