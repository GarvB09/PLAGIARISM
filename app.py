"""
AI-Powered Plagiarism & Authorship Detection System
Flask Backend Server
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime

from utils.features import extract_features
from utils.text_analysis import calculate_plagiarism_score
from utils.scoring import calculate_ai_score, calculate_authorship_score, generate_conclusion

app = Flask(__name__)
CORS(app)

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "AI-Plagiarism-Authorship-Analyzer (Python)",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.route("/api/features", methods=["GET"])
def features_info():
    return jsonify({
        "features": [
            {
                "name": "avg_sentence_length",
                "description": "Average words per sentence",
                "ai_indicator": "Uniform 15–25 words",
                "human_indicator": "Varied 10–30+ words"
            },
            {
                "name": "vocabulary_diversity",
                "description": "Unique words / total words (Type-Token Ratio)",
                "ai_indicator": "Low (< 0.5)",
                "human_indicator": "High (> 0.65)"
            },
            {
                "name": "transition_word_freq",
                "description": "Frequency of connector words (however, thus, etc.)",
                "ai_indicator": "High (> 0.03)",
                "human_indicator": "Low (< 0.02)"
            },
            {
                "name": "repetition_score",
                "description": "Word repetition rate",
                "ai_indicator": "Low (< 0.15)",
                "human_indicator": "High (> 0.20)"
            },
            {
                "name": "perplexity_score",
                "description": "Structural predictability (sentence length variance)",
                "ai_indicator": "High (> 0.7)",
                "human_indicator": "Low (< 0.5)"
            }
        ],
        "scoring_guide": {
            "ai_score": {
                "0-34": "Human-written",
                "35-64": "Mixed signals / possibly AI-assisted",
                "65-100": "Likely AI-generated"
            },
            "plagiarism_score": {
                "0-30": "Original content",
                "30-60": "Moderate similarity",
                "60-100": "High plagiarism risk"
            },
            "authorship_score": {
                "70-100": "Match — consistent author",
                "40-69": "Partial — some style variation",
                "0-39": "Mismatch — likely different author"
            }
        }
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: 'text'"}), 400

    text = data.get("text", "").strip()
    reference_text = data.get("referenceText", "").strip()
    user_history = data.get("userHistory", "").strip()

    if len(text) < 20:
        return jsonify({"error": "Text is too short. Please provide at least 20 characters."}), 400

    # ── AI Detection ──────────────────────────────────────────────────────────
    ai_score, ai_confidence, ai_reasons, features = calculate_ai_score(text)

    # ── Plagiarism Detection ──────────────────────────────────────────────────
    plagiarism_score = None
    jaccard_score = None
    cosine_score = None
    plagiarism_details = None

    if reference_text:
        jaccard_score, cosine_score, plagiarism_score = calculate_plagiarism_score(text, reference_text)
        plagiarism_details = {
            "jaccard_similarity": jaccard_score,
            "cosine_similarity": cosine_score,
            "combined_score": plagiarism_score
        }

    # ── Authorship Detection ──────────────────────────────────────────────────
    authorship_score = None
    authorship_verdict = None
    authorship_reasons = None

    if user_history:
        authorship_score, authorship_verdict, authorship_reasons = calculate_authorship_score(text, user_history)

    # ── Conclusion ────────────────────────────────────────────────────────────
    conclusion = generate_conclusion(ai_score, plagiarism_score, authorship_score)

    # ── All Reasons ───────────────────────────────────────────────────────────
    all_reasons = ai_reasons.copy()
    if authorship_reasons:
        all_reasons.extend(authorship_reasons)

    return jsonify({
        "ai_score": ai_score,
        "ai_confidence": ai_confidence,
        "plagiarism_score": plagiarism_score,
        "plagiarism_details": plagiarism_details,
        "authorship_score": authorship_score,
        "authorship_verdict": authorship_verdict,
        "authorship_reasons": authorship_reasons,
        "features": features,
        "conclusion": conclusion,
        "reasons": all_reasons
    })


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 AI Plagiarism Detector running at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
