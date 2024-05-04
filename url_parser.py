import re
from urllib.parse import urlparse, parse_qs

def extract_uci_url_features(url):
    features = {}
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    # Extract features
    features['URLLength'] = len(url)
    features['DomainLength'] = len(domain)
    features['IsDomainIP'] = int(bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain)))
    features['TLD'] = parsed_url.hostname.split('.')[-1] if parsed_url.hostname else ''
    features['TLDLength'] = len(features['TLD'])
    del features['TLD']
    features['NoOfSubDomain'] = len(domain.split('.')) - 2
    features['HasObfuscation'] = int('//' in path or '@' in url)
    features['IsHTTPS'] = int(parsed_url.scheme == 'https')
    features['NoOfEqualsInURL'] = url.count('=')
    features['NoOfQMarkInURL'] = url.count('?')
    features['NoOfAmpersandInURL'] = url.count('&')
    features['NoOfOtherSpecialCharsInURL'] = sum(map(url.count, ['#', '$', '%', '^', '*', '(', ')', '+']))

    # Calculate character and digit ratios
    total_chars = len(url)
    features['NoOfLettersInURL'] = len(re.findall('[a-zA-Z]', url))
    features['LetterRatioInURL'] = features['NoOfLettersInURL'] / total_chars if total_chars > 0 else 0
    features['NoOfDegitsInURL'] = len(re.findall('\d', url))
    features['DegitRatioInURL'] = features['NoOfDegitsInURL'] / total_chars if total_chars > 0 else 0
    features['SpacialCharRatioInURL'] = features['NoOfOtherSpecialCharsInURL'] / total_chars if total_chars > 0 else 0

    return features

def extract_kaggle_url_features(url):
    # Normalize URL by removing the scheme (http:// or https://)
    normalized_url = url.replace('http://', '').replace('https://', '')
    parsed_url = urlparse('//' + normalized_url)  # Prepend '//' to make urlparse work correctly

    # Parsing query parameters
    query_params = parse_qs(parsed_url.query)

    # Prepare to extract features
    hostname = parsed_url.hostname if parsed_url.hostname else ""
    path = parsed_url.path if parsed_url.path else ""
    digits_in_url = sum(c.isdigit() for c in url)
    digits_in_hostname = sum(c.isdigit() for c in hostname)

    # Extract features
    features = {
        "length_url": len(url),
        "length_hostname": len(hostname),
        "ip": int(parsed_url.hostname.replace('.', '').isdigit()) if parsed_url.hostname else 0,
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": url.count("@"),
        "nb_qm": url.count("?"),
        "nb_and": url.count("&"),
        "nb_or": url.count("|"),
        "nb_eq": len(query_params),
        "nb_underscore": url.count("_"),
        "nb_tilde": url.count("~"),
        "nb_percent": url.count("%"),
        "nb_slash": url.count("/"),
        "nb_star": url.count("*"),
        "nb_colon": url.count(":"),
        "nb_comma": url.count(","),
        "nb_semicolumn": url.count(";"),
        "nb_dollar": url.count("$"),
        "nb_space": url.count(" "),
        "nb_www": hostname.count("www"),
        "nb_com": hostname.count(".com"),
        "nb_dslash": url.count("//"),
        "http_in_path": 1 if "http" in path else 0,
        "https_token": 1 if "https" in hostname else 0,
        "ratio_digits_url": digits_in_url / len(url) if url else 0,
        "ratio_digits_host": digits_in_hostname / len(hostname) if hostname else 0,
        "punycode": 1 if 'xn--' in hostname else 0
    }

    return features