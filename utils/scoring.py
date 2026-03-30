"""
Scoring Module
AI detection scoring and authorship/stylometry analysis.
"""

from typing import Dict, List, Tuple
from utils.features import extract_features


# ─── AI DETECTION THRESHOLDS ────────────────────────────────────────────────

AI_THRESHOLDS = {
    "sentence_length_min": 15,
    "sentence_length_max": 25,
    "low_vocabulary_diversity": 0.5,
    "high_transition_freq": 0.03,
    "low_repetition": 0.15,
    "high_perplexity": 0.7,
}

AI_WEIGHTS = {
    "sentence_length": 25,
    "vocabulary_diversity": 20,
    "transition_words": 20,
    "repetition": 15,
    "perplexity": 20,
}


def calculate_ai_score(text: str) -> Tuple[float, str, List[str], Dict]:
    """
    Returns (score 0-100, confidence, reasons, features).
    Higher score = more likely AI-generated.
    """
    features = extract_features(text)
    score = 0
    reasons = []

    # 1. Sentence length (AI tends toward uniform 15-25 words)
    sl = features["avg_sentence_length"]
    if AI_THRESHOLDS["sentence_length_min"] <= sl <= AI_THRESHOLDS["sentence_length_max"]:
        score += AI_WEIGHTS["sentence_length"]
        reasons.append(f"Uniform sentence length ({sl:.1f} words avg — typical of AI)")
    else:
        reasons.append(f"Varied sentence length ({sl:.1f} words avg — human-like)")

    # 2. Vocabulary diversity (AI tends to be lower)
    vd = features["vocabulary_diversity"]
    if vd < AI_THRESHOLDS["low_vocabulary_diversity"]:
        score += AI_WEIGHTS["vocabulary_diversity"]
        reasons.append(f"Low vocabulary diversity ({vd:.2f} — AI indicator)")
    else:
        reasons.append(f"Good vocabulary diversity ({vd:.2f} — human-like)")

    # 3. Transition word frequency (AI overuses connectors)
    tf = features["transition_word_freq"]
    if tf > AI_THRESHOLDS["high_transition_freq"]:
        score += AI_WEIGHTS["transition_words"]
        reasons.append(f"High use of transition words ({tf:.3f} — AI pattern)")
    else:
        reasons.append(f"Natural transition word usage ({tf:.3f})")

    # 4. Repetition score (AI tends to repeat less)
    rs = features["repetition_score"]
    if rs < AI_THRESHOLDS["low_repetition"]:
        score += AI_WEIGHTS["repetition"]
        reasons.append(f"Low word repetition ({rs:.2f} — AI tends to avoid repeating)")
    else:
        reasons.append(f"Natural word repetition ({rs:.2f} — human-like)")

    # 5. Perplexity score (AI has more uniform sentence structure)
    ps = features["perplexity_score"]
    if ps > AI_THRESHOLDS["high_perplexity"]:
        score += AI_WEIGHTS["perplexity"]
        reasons.append(f"High structural uniformity ({ps:.2f} — AI indicator)")
    else:
        reasons.append(f"Varied sentence structure ({ps:.2f} — human-like)")

    # Confidence level
    if score < 35:
        confidence = "Low"
    elif score < 65:
        confidence = "Medium"
    else:
        confidence = "High"

    return round(score, 2), confidence, reasons, features


# ─── AUTHORSHIP DETECTION ────────────────────────────────────────────────────

def calculate_authorship_score(text: str, reference: str) -> Tuple[float, str, List[str]]:
    """
    Compares stylometric features between two texts.
    Returns (score 0-100, verdict, reasons).
    Higher score = more similar writing style.
    """
    features1 = extract_features(text)
    features2 = extract_features(reference)

    feature_keys = [
        "avg_sentence_length",
        "vocabulary_diversity",
        "transition_word_freq",
        "repetition_score",
        "perplexity_score",
    ]

    similarities = []
    reasons = []

    tolerances = {
        "avg_sentence_length": 5.0,
        "vocabulary_diversity": 0.15,
        "transition_word_freq": 0.015,
        "repetition_score": 0.15,
        "perplexity_score": 0.2,
    }

    labels = {
        "avg_sentence_length": "Sentence length",
        "vocabulary_diversity": "Vocabulary diversity",
        "transition_word_freq": "Transition word usage",
        "repetition_score": "Word repetition patterns",
        "perplexity_score": "Structural consistency",
    }

    for key in feature_keys:
        v1 = features1[key]
        v2 = features2[key]
        tol = tolerances[key]
        diff = abs(v1 - v2)
        sim = max(0.0, 1.0 - (diff / tol))
        similarities.append(min(sim, 1.0))

        label = labels[key]
        if sim > 0.7:
            reasons.append(f"✓ {label} matches ({v1:.3f} vs {v2:.3f})")
        else:
            reasons.append(f"✗ {label} differs ({v1:.3f} vs {v2:.3f})")

    score = (sum(similarities) / len(similarities)) * 100

    if score > 80:
        verdict = "Match"
    elif score > 40:
        verdict = "Partial Match"
    else:
        verdict = "Mismatch"

    return round(score, 2), verdict, reasons


# ─── CONCLUSION GENERATION ───────────────────────────────────────────────────

def generate_conclusion(ai_score: float, plagiarism_score: float = None, authorship_score: float = None) -> str:
    parts = []

    if ai_score >= 65:
        parts.append("Text appears to be AI-generated")
    elif ai_score >= 35:
        parts.append("Text may be AI-assisted or partially AI-generated")
    else:
        parts.append("Text appears to be human-written")

    if plagiarism_score is not None:
        if plagiarism_score >= 60:
            parts.append("high plagiarism risk detected")
        elif plagiarism_score >= 30:
            parts.append("moderate similarity to reference text")
        else:
            parts.append("appears to be original content")

    if authorship_score is not None:
        if authorship_score >= 70:
            parts.append("writing style matches the reference author")
        elif authorship_score >= 40:
            parts.append("some stylistic similarities with reference author")
        else:
            parts.append("writing style does not match reference author")

    return ". ".join(parts) + "."
