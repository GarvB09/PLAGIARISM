"""
Text Analysis Module
Implements Jaccard and Cosine similarity algorithms for plagiarism detection.
"""

import math
import re
from collections import Counter
from typing import Tuple


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Jaccard Similarity: |A ∩ B| / |A ∪ B|
    Good for direct word overlap detection.
    """
    words1 = set(tokenize(text1))
    words2 = set(tokenize(text2))

    if not words1 and not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def cosine_similarity(text1: str, text2: str) -> float:
    """
    Cosine Similarity: dot(A, B) / (|A| * |B|)
    Better for paraphrased content — considers word frequency.
    """
    words1 = tokenize(text1)
    words2 = tokenize(text2)

    if not words1 or not words2:
        return 0.0

    freq1 = Counter(words1)
    freq2 = Counter(words2)

    all_words = set(freq1.keys()) | set(freq2.keys())

    dot_product = sum(freq1.get(w, 0) * freq2.get(w, 0) for w in all_words)
    magnitude1 = math.sqrt(sum(v ** 2 for v in freq1.values()))
    magnitude2 = math.sqrt(sum(v ** 2 for v in freq2.values()))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def calculate_plagiarism_score(text: str, reference: str) -> Tuple[float, float, float]:
    """
    Returns (jaccard, cosine, combined_score) all as 0-100 percentages.
    Combined = weighted average: 60% jaccard + 40% cosine
    """
    jaccard = jaccard_similarity(text, reference)
    cosine = cosine_similarity(text, reference)
    combined = (0.6 * jaccard + 0.4 * cosine) * 100

    return round(jaccard * 100, 2), round(cosine * 100, 2), round(combined, 2)
