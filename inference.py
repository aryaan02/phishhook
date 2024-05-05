import torch

from url_parser import extract_uci_url_features
from phishhooknet import PhishHookNet

model = PhishHookNet(input_size=16)
model.load_state_dict(torch.load("phishing_uci_url_model.pth"))
model.eval()

def infer(url):
    features = extract_uci_url_features(url)
    feature_values = list(features.values())
    input_tensor = torch.tensor([feature_values], dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor).round().item()
        predicted_class = "Yes" if output == 1 else "No"
        print(f"Predicted class: {predicted_class}, Probability: {output}")
        return {"probability": output, "is_phishing": predicted_class}
