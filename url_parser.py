from urllib.parse import urlparse, parse_qs

def extract_url_features(url):
    features = {}
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # Basic features from URL components
    features['length_url'] = len(url)
    features['length_hostname'] = len(parsed_url.netloc)
    features['ip'] = 1 if parsed_url.hostname and parsed_url.hostname.replace('.', '').isdigit() else 0
    features['nb_dots'] = url.count('.')
    features['nb_hyphens'] = url.count('-')
    features['nb_at'] = url.count('@')
    features['nb_qm'] = url.count('?')
    features['nb_and'] = url.count('&')
    features['nb_or'] = url.count('|')
    features['nb_eq'] = len(query_params)
    features['nb_underscore'] = url.count('_')
    features['nb_tilde'] = url.count('~')
    features['nb_percent'] = url.count('%')
    features['nb_slash'] = url.count('/')
    features['nb_star'] = url.count('*')
    features['nb_colon'] = url.count(':')
    features['nb_comma'] = url.count(',')
    features['nb_semicolumn'] = url.count(';')
    features['nb_dollar'] = url.count('$')
    features['nb_space'] = url.count(' ')
    features['nb_dslash'] = url.count('//')

    return features

