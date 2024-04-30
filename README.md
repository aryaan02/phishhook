# PhishHook - A Phishing URL Detector

PhishHook is a Chrome Extension used to detect phishing URLs. It uses a machine learning model to classify URLs as phishing or not phishing when the user hovers over a link on a webpage. The model was trained on a dataset of phishing URLs and non-phishing URLs. The extension also provides a popup with information about the URL and a warning if it is classified as phishing.

## Reproduce the Model

1. Clone the repository
2. Run `pip install -r requirements.txt`
3. Open phishhook.ipynb in Jupyter Notebook
4. Run the notebook to train the model

## Dataset

The dataset used to train the model is from [Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls). It contains 507,195 URLs, each labeled as "good" or "bad". The dataset was preprocessed and features were extracted from the URLs to train the model.
