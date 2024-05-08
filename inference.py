import torch

from url_parser import extract_url_features
from phishhooknet import PhishHookNet

# Load the model
model = PhishHookNet(input_size=16)
model.load_state_dict(torch.load("phishing_url_model.pth"))
model.eval()


def infer(url):
    """
    Given a URL, infer if it is a phishing URL or not.

    Args:
        url (str): The URL to infer.

    Returns:
        dict: A dictionary containing the probability of the URL being a phishing URL and the predicted class.
    """
    # Extract features from the URL
    features = extract_url_features(url)
    feature_values = list(features.values())
    input_tensor = torch.tensor([feature_values], dtype=torch.float32)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor).item()
        predicted_class = "Yes" if output > 0.5 else "No"

    return {"probability": output, "is_phishing": predicted_class}
