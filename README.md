# PlagScan AI — Plagiarism & Authorship Detection System

A full-stack AI-powered text analysis system built with **Python + Flask**. Detects AI-generated content using a trained machine learning model, checks plagiarism via Jaccard/Cosine similarity and the Copyleaks API, and performs authorship stylometry analysis.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Detection** | GradientBoosting ML model trained on 2,750 human vs AI essays. AUC = 0.9999, 10-fold CV AUC = 0.9997 |
| 🔍 **Plagiarism Detection** | Jaccard + Cosine similarity against a reference text |
| 🌐 **Internet Scan** | Copyleaks API integration — scans against billions of web pages |
| 📂 **Document Corpus** | Upload `.txt`, `.pdf`, `.docx` files and compare any text against all of them |
| ✍️ **Authorship Analysis** | Stylometric comparison of writing style between two texts |

---

## 🧠 ML Model

Trained on the [Human vs AI Generated Essays](https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays) dataset from Kaggle.

**Model:** GradientBoosting Classifier on 10 linguistic features (no API keys, no external calls — runs fully offline).

### Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | **0.9999** |
| 10-Fold CV AUC | **0.9997 ± 0.0005** |
| Accuracy | 99.27% |
| F1 Score | 99.27% |
| Avg Precision | 99.99% |
| Training samples | 2,200 |
| Test samples | 550 |

### Feature Importance

| Feature | Importance | What it captures |
|---|---|---|
| `sentence_cv` | **57.8%** | Coefficient of variation of sentence lengths — AI is uniform, humans vary |
| `perplexity_score` | 19.4% | Structural predictability — AI is highly regular |
| `sentence_length_variance` | 11.9% | Raw variance of sentence lengths |
| `avg_word_length` | 8.3% | AI favours longer, formal vocabulary |
| `vocabulary_diversity` | 1.6% | Type-token ratio |
| others | 1.0% | Transition words, punctuation, uppercase, repetition |

### All Models Trained

| Model | AUC | 10-CV AUC | F1 | Type |
|---|---|---|---|---|
| **GradientBoosting** ← selected | **0.9999** | **0.9997 ± 0.0005** | 0.9927 | linguistic |
| RandomForest | 0.9998 | 0.9998 ± 0.0003 | 0.9945 | linguistic |
| Ensemble (GB+RF+LR) | 0.9995 | 0.9996 ± 0.0006 | 0.9927 | linguistic |
| Logistic Regression | 0.9982 | 0.9984 ± 0.0019 | 0.9891 | linguistic |
| Logistic + TF-IDF | 0.9979 | — | 0.9964 | combined |
| LinearSVM | 0.9975 | 0.9981 ± 0.0022 | 0.9855 | linguistic |

---

## 📁 Project Structure

```
PLAGIARISM/
├── app.py                        # Flask server — all API routes
├── requirements.txt              # Python dependencies
├── README.md
├── templates/
│   └── index.html                # Web UI (3 tabs: Analyze / Internet Scan / Corpus)
├── utils/
│   ├── features.py               # Heuristic linguistic feature extractor
│   ├── text_analysis.py          # Jaccard + Cosine similarity algorithms
│   ├── scoring.py                # Heuristic AI scoring + authorship detection
│   ├── copyleaks_client.py       # Copyleaks API async integration
│   └── corpus_manager.py        # Local document corpus (txt/pdf/docx)
├── ml/
│   ├── feature_extractor.py      # 10 linguistic features + TF-IDF extractor
│   ├── train.py                  # Full training pipeline (run to retrain)
│   └── predictor.py              # Load model + run inference
└── models/
    ├── best_model.pkl            # Trained GradientBoosting model
    ├── extractor.pkl             # Fitted TF-IDF feature extractor
    └── metadata.json            # Model metrics + feature names
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/GarvB09/PLAGIARISM.git
cd PLAGIARISM

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py

# 5. Open browser
# http://localhost:5000
```

The ML model is pre-trained and included in `models/` — no training step needed to run.

---

## 🔁 Retrain the Model

To retrain on new data or the original Kaggle dataset:

```bash
# Download dataset from:
# https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays

python ml/train.py --data path/to/essays.csv

# Optional flags:
# --output models      output directory (default: models)
# --test 0.2           test split ratio (default: 0.2)
```

The training pipeline automatically:
- Detects and strips leftover AI prompt prefixes
- Chunks long human essays to match AI essay length (fixes length bias)
- Trains 6 candidate models and picks the best by AUC
- Runs 10-fold cross-validation
- Tunes the classification threshold via precision-recall curve
- Saves model, extractor, and full metrics to `models/`

---

## 📡 API Reference

### `POST /api/analyze`
Main analysis endpoint — AI detection + optional plagiarism + optional authorship.

**Request:**
```json
{
  "text": "Text to analyze...",
  "referenceText": "(optional) source text for plagiarism comparison",
  "userHistory":   "(optional) known writing samples for authorship check"
}
```

**Response:**
```json
{
  "ai_score": 91.4,
  "ai_confidence": "High",
  "ai_verdict": "AI-generated",
  "ai_method": "ml+heuristic",
  "plagiarism_score": 42.5,
  "plagiarism_details": {
    "jaccard_similarity": 38.0,
    "cosine_similarity": 49.0,
    "combined_score": 42.5
  },
  "authorship_score": 85.0,
  "authorship_verdict": "Match",
  "features": { "avg_sentence_length": 18.3, "sentence_cv": 0.04, "..." : "..." },
  "conclusion": "Text appears to be AI-generated.",
  "reasons": ["..."]
}
```

### `POST /api/copyleaks/scan`
Submit text to Copyleaks for internet-wide plagiarism scan (async).

**Request:** `{ "text": "..." }`
**Response:** `{ "scan_id": "abc123", "status": "pending" }`

### `GET /api/copyleaks/result/<scan_id>`
Poll for Copyleaks results after submitting a scan.

### `POST /api/corpus/upload`
Upload a `.txt`, `.pdf`, or `.docx` file to the local corpus (multipart/form-data, key=`file`).

### `GET /api/corpus`
List all documents currently in the corpus.

### `POST /api/corpus/compare`
Compare text against all documents in the corpus.

**Request:** `{ "text": "..." }`

### `DELETE /api/corpus/delete/<doc_id>`
Remove a document from the corpus.

### `GET /api/health`
Returns server status, ML model info, and corpus document count.

---

## 🌐 Copyleaks Setup (Optional)

Copyleaks scans text against the live internet. It is **async** — results arrive via webhook.

```bash
# Set environment variables before running app.py
export COPYLEAKS_EMAIL="your@email.com"
export COPYLEAKS_API_KEY="your-api-key"
export COPYLEAKS_SANDBOX="true"        # false in production
export WEBHOOK_BASE_URL="https://your-public-url.com"
```

For local development, expose port 5000 publicly using [ngrok](https://ngrok.com):
```bash
ngrok http 5000
# Then set WEBHOOK_BASE_URL=https://xxxx.ngrok.io
```

Sign up free at [copyleaks.com](https://copyleaks.com) (~20 pages/month free).

---

## 📊 Scoring Guide

### AI Detection Score (0–100)

| Score | Verdict | Meaning |
|---|---|---|
| 0–39 | ✅ Human-written | Natural writing patterns |
| 40–64 | ⚠️ Possibly AI-assisted | Mixed signals |
| 65–100 | ❌ AI-generated | Strong AI indicators |

> For texts under 40 words the system uses heuristic scoring only.
> For 40+ words it blends ML probability with heuristic (ML weight increases with text length up to 85%).

### Plagiarism Score (0–100)

| Score | Verdict |
|---|---|
| 0–29 | ✅ Original content |
| 30–59 | ⚠️ Moderate similarity |
| 60–100 | ❌ High plagiarism risk |

### Authorship Score (0–100)

| Score | Verdict |
|---|---|
| 70–100 | ✅ Match — same author likely |
| 40–69 | ⚠️ Partial match |
| 0–39 | ❌ Mismatch — likely different author |

---

## 🧪 Test with cURL

```bash
# AI detection
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Furthermore, machine learning algorithms demonstrate remarkable capabilities in pattern recognition. Moreover, AI integration significantly improves operational efficiency across various sectors."}'

# With plagiarism check
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog",
    "referenceText": "The quick brown fox jumps over the lazy dog"
  }'

# Health check
curl http://localhost:5000/api/health
```

---

## 📦 Dependencies

```
flask==3.0.3
flask-cors==4.0.1
requests==2.32.3
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
pypdf==4.3.1
python-docx==1.1.2
```

---

## 🔐 Security Notes

- No data persistence — all analysis is stateless
- Corpus is in-memory only (cleared on server restart)
- No external API calls except Copyleaks (optional)
- Can run entirely offline for AI detection + corpus comparison

---

## 📝 Limitations

- **ML model** works best on texts of 100+ words. Short texts fall back to heuristic scoring.
- **Copyleaks** requires a public webhook URL — use ngrok for local development.
- **Corpus** is in-memory and resets when the server restarts. For persistence, use a database.
- **Language** — all features are optimised for English text.

---

## 👥 Project Info

**Built for:** College Capstone Project  
**Stack:** Python · Flask · scikit-learn · GradientBoosting · TF-IDF  
**Dataset:** [Human vs AI Generated Essays — Kaggle](https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays)  
**Version:** 4.0.0
