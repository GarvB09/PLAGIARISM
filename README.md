# 🎓 AI-Powered Plagiarism & Authorship Detection System — Python Edition

A Python/Flask rewrite of the original React+Express project.

## 📁 Project Structure

```
plagiarism_detector/
├── app.py                  # Flask server (main entry point)
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web UI
└── utils/
    ├── features.py         # 5-feature linguistic extraction
    ├── text_analysis.py    # Jaccard + Cosine similarity
    └── scoring.py          # AI detection & authorship scoring
```

## 🚀 Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py

# 4. Open your browser
# http://localhost:5000
```

## 📡 API Endpoints

### POST /api/analyze
```json
{
  "text": "Your text here...",
  "referenceText": "(optional) source text for plagiarism check",
  "userHistory": "(optional) known writing samples for authorship check"
}
```

**Response:**
```json
{
  "ai_score": 75,
  "ai_confidence": "High",
  "plagiarism_score": 42.5,
  "plagiarism_details": {
    "jaccard_similarity": 38.0,
    "cosine_similarity": 49.0,
    "combined_score": 42.4
  },
  "authorship_score": 85.0,
  "authorship_verdict": "Match",
  "features": { ... },
  "conclusion": "...",
  "reasons": ["..."]
}
```

### GET /api/health
### GET /api/features

## 🧪 Test with cURL

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has revolutionized problem-solving. Machine learning demonstrates remarkable capabilities. Furthermore, AI integration improves efficiency significantly.",
    "referenceText": "AI has changed how we solve problems. Machine learning shows impressive abilities."
  }'
```

## 📊 Scoring Guide

| Score | AI Detection | Plagiarism | Authorship |
|-------|-------------|------------|------------|
| Low   | ✅ Human-written | ✅ Original | ✗ Mismatch |
| Mid   | ⚠ Mixed signals | ⚠ Moderate | ⚠ Partial |
| High  | ✗ AI-generated | ✗ High risk | ✅ Match |
