# PhishHook - A Phishing URL Detector

PhishHook is a Chrome Extension used to detect phishing URLs. It uses a machine learning model to classify URLs as phishing or not phishing when the user hovers over a link on a webpage. The model was trained on a dataset of phishing URLs and non-phishing URLs. The extension also provides a popup with information about the URL and a warning if it is classified as phishing.

## Dataset

The dataset used to train the model is from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) and contains 134,850 legitimate URLs and 100,945 phishing URLs.

## Reproduce the Model

1. Clone the repository
2. Create a virtual environment by running `python -m venv env`
3. Activate the virtual environment by running `source env/bin/activate` on Mac/Linux or `.\env\Scripts\activate` on Windows
4. Run `pip install -r requirements.txt`
5. Open phishhook_train.ipynb in Jupyter Notebook
6. Run the notebook to train the model

## Chrome Extension

Here are the steps to load the extension in Chrome:
1. Open Chrome and go to chrome://extensions/
2. Turn on Developer mode
3. Click on Load unpacked
4. Select the extension folder

Then, start the Flask server by running `python app.py` in the terminal.

Now, the extension should be loaded in Chrome. To test the extension, go to a webpage with links and hover over any red links. A popup should appear with information about the URL and a warning if it is classified as phishing.

You can also access the extensions in the toolbar by clicking on the puzzle icon and pinning the PhishHook extension. Clicking on the extension icon will show the popup with information about all the URLs in the current web page.
