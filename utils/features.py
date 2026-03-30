"""
Feature Extraction Module
Extracts 5 key linguistic features from text for AI detection, plagiarism, and authorship analysis.
"""

import re
from collections import Counter
from typing import Dict


TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "nevertheless",
    "consequently", "additionally", "subsequently", "alternatively",
    "thus", "hence", "accordingly", "meanwhile", "nonetheless",
    "in conclusion", "in addition", "on the other hand", "as a result",
    "for instance", "for example", "in contrast", "in summary",
    "to summarize", "in particular", "specifically", "notably"
}


def tokenize_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.strip()) > 0]


def tokenize_words(text: str) -> list[str]:
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def extract_features(text: str) -> Dict[str, float]:
    """
    Extract 5 linguistic features from text:
    1. avg_sentence_length   - Average words per sentence
    2. vocabulary_diversity  - Unique words / total words (Type-Token Ratio)
    3. transition_word_freq  - Frequency of transition/connector words
    4. repetition_score      - Word repetition rate
    5. perplexity_score      - Structure predictability (sentence length variance-based)
    """
    words = tokenize_words(text)
    sentences = tokenize_sentences(text)

    if not words or not sentences:
        return {
            "avg_sentence_length": 0,
            "vocabulary_diversity": 0,
            "transition_word_freq": 0,
            "repetition_score": 0,
            "perplexity_score": 0,
        }

    # 1. Average sentence length
    sentence_lengths = [len(tokenize_words(s)) for s in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

    # 2. Vocabulary diversity (Type-Token Ratio)
    unique_words = set(words)
    vocabulary_diversity = len(unique_words) / len(words)

    # 3. Transition word frequency
    text_lower = text.lower()
    transition_count = sum(1 for tw in TRANSITION_WORDS if tw in text_lower)
    transition_word_freq = transition_count / len(words)

    # 4. Repetition score (word repetition rate)
    word_counts = Counter(words)
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    repetition_score = repeated_words / len(unique_words) if unique_words else 0

    # 5. Perplexity score (based on sentence length variance — uniform = high perplexity)
    if len(sentence_lengths) > 1:
        mean_len = avg_sentence_length
        variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        # Normalize: low variance → high perplexity (AI-like), high variance → low perplexity (human-like)
        normalized_variance = min(variance / 100, 1.0)
        perplexity_score = 1.0 - normalized_variance
    else:
        perplexity_score = 0.5

    return {
        "avg_sentence_length": round(avg_sentence_length, 2),
        "vocabulary_diversity": round(vocabulary_diversity, 4),
        "transition_word_freq": round(transition_word_freq, 4),
        "repetition_score": round(repetition_score, 4),
        "perplexity_score": round(perplexity_score, 4),
    }
