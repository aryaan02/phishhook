import re
from urllib.parse import urlparse


def extract_url_features(url):
    """
    Extracts features from a URL for phishing detection (UCI Dataset).

    Args:
        url (str): The URL to extract features from.

    Returns:
        dict: A dictionary containing the extracted features.
    """
    # Parse the URL
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    # Clean the URL by removing the scheme (http:// or https://)
    cleaned_url = url.replace("http://www.", "").replace("https://www.", "")

    # Extract features
    features = {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": int(
            bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain))
        ),
        "TLDLength": (
            len(parsed_url.hostname.split(".")[-1]) if parsed_url.hostname else 0
        ),
        "NoOfSubDomain": len(domain.split(".")) - 2,
        "HasObfuscation": int("//" in path or "@" in url),
        "IsHTTPS": int(parsed_url.scheme == "https"),
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": sum(
            url.count(c) for c in [";", ":", "@", "%", "-", "_", "~"]
        ),
        "SpacialCharRatioInURL": (
            sum(url.count(c) for c in [";", ":", "@", "%", "-", "_", "~"]) / len(url)
            if len(url) > 0
            else 0
        ),
        "NoOfLettersInURL": sum(c.isalpha() for c in cleaned_url),
        "LetterRatioInURL": (
            sum(c.isalpha() for c in cleaned_url) / len(url) if len(url) > 0 else 0
        ),
        "NoOfDegitsInURL": sum(c.isdigit() for c in url),
        "DegitRatioInURL": (
            sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
        ),
    }

    return features
