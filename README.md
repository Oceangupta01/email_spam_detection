# Email / SMS Spam Detection System

A machine-learning app that classifies emails and SMS messages as **Spam**, **Suspicious** or **Not Spam**. It explains each verdict word by word and flags phishing risk signals. The multipage Streamlit dashboard runs on Streamlit Community Cloud with no extra configuration.

**Author:** Ocean Gupta, GCET Jammu

## Features

- **Spam Detector**: paste any message, including a raw email with headers or HTML. You get a calibrated spam probability, gauges, word-level highlighting, the top contributing features, and rule-based risk signals: phishing, urgency, links, money and pharma/gambling offers. Session history is kept per browser session.
- **Batch Analysis**: upload a CSV/TXT (or use the demo set) to score thousands of messages, filter them, chart the results and download them as CSV.
- **Model Performance**: held-out test metrics, the original model vs the improved one, confusion matrix, ROC and precision-recall curves, threshold tuning, a five-algorithm comparison, learned weights and the hardest mistakes.
- **Dataset Insights**: class balance, message lengths, tell-tale features, vocabulary charts and word clouds.
- **How It Works**: the pipeline, the root causes of the original "everything is Not Spam" bug, and the tech stack.
- **Adjustable threshold** in the sidebar, with live precision/recall at that threshold.
- **FastAPI REST endpoint** (`api.py`) for programmatic use.

## Results (held-out test set, 1,939 messages)

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Original pipeline | 84.6% | 76.2% | 57.3% | 65.4% | 0.909 |
| **Improved model** | **97.5%** | **95.1%** | **95.1%** | **95.1%** | **0.995** |

Email spam F1 rose from **55.1% to 96.4%**. The original model was trained on SMS texts only and deleted digits, symbols and links, so it labelled most real email spam as "Not Spam".

## Project structure

```
app.py                  Streamlit entry point (multipage navigation)
app_pages/              Detector, Batch, Performance, Insights, About pages
ui/                     Theme, CSS and shared components
spam_model/             SpamDetector, explanations, rule-based risk signals
data_preprocessing.py   Text normalisation, signal tokens, email parsing
train.py                Training + evaluation pipeline -> model.pkl, vectorizer.pkl, metrics.json
predict.py              Artifact loading and prediction helpers
api.py                  FastAPI service
data/                   spam.csv (SMS Spam Collection), emails.csv.gz (SpamAssassin)
scripts/                Script that rebuilds data/emails.csv.gz from the SpamAssassin corpus
tests/                  pytest suite
.streamlit/config.toml  Theme and server settings
```

## Model

- **Data:** SMS Spam Collection (5,169 unique messages) plus the SpamAssassin public email corpus (4,526 unique emails), deduplicated, with a stratified 80/20 split.
- **Preprocessing:** emails are parsed (From, Subject, body; HTML stripped). Links, email addresses, money amounts, percentages, phone numbers, ALL-CAPS words and `!` become tokens instead of being deleted. After that the text is lowercased, stop words are removed and words are Porter-stemmed.
- **Features:** TF-IDF over words and word pairs (50,000 features, sublinear TF), fitted on the training split only.
- **Classifier:** `CalibratedClassifierCV(LinearSVC(class_weight="balanced"))`, which gives reliable probabilities.
- **Rule layer:** phishing patterns and strong spam signals turn a borderline message into **Suspicious**.

## Run locally

Requires Python 3.10+ (3.12 recommended).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The trained artifacts (`model.pkl`, `vectorizer.pkl`, `metrics.json`) are committed. If they are missing or were built with a different scikit-learn version, the app retrains automatically on first start (about 15 s).

To retrain manually and print the evaluation table:

```bash
python train.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and choose **Create app → Deploy a public app from GitHub**.
3. Repository `Oceangupta01/email_spam_detection`, branch `main`, main file path **`app.py`**.
4. Under **Advanced settings**, choose Python 3.12, then click **Deploy**.

Every push to `main` redeploys the app automatically. You don't need a Dockerfile, Procfile or setup script; Streamlit Cloud installs `requirements.txt` and reads `.streamlit/config.toml`. `scikit-learn` is pinned so the committed model files load without retraining.

## REST API

```bash
pip install -r requirements-dev.txt
uvicorn api:app --port 8000
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "WINNER!! Claim your $1000 prize now at http://win.example"}'
# {"label": "spam", "confidence": 0.99}
```

Interactive docs are at http://localhost:8000/docs.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Tech stack

Python, scikit-learn, NLTK (Porter stemmer), pandas, NumPy, Streamlit, Plotly, WordCloud, FastAPI.

## License

MIT
