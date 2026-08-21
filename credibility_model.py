"""
credibility_model.py
Author : Akshay Shukla
Modified for: Domain Calibration & Streamlit Integration
"""

import os
import joblib
import numpy as np

from preprocess import preprocess_text
from similarity import SimilarityChecker


class CredibilityAnalyzer:

    def __init__(self, model_path="fake_news_model.joblib", vectorizer_path="vectorizer.joblib", trusted_news_path="trusted_news.csv"):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.similarity_checker = SimilarityChecker(trusted_news_path=trusted_news_path)

    def analyze(self, article: str) -> dict:
        if not article or not isinstance(article, str):
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "similarity_score": 0.0,
                "credibility_score": 50.0,
                "credibility_level": "Unverified",
                "matched_article": ""
            }

        # 1. Preprocess
        cleaned_article = preprocess_text(article)
        if not cleaned_article.strip():
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "similarity_score": 0.0,
                "credibility_score": 50.0,
                "credibility_level": "Unverified",
                "matched_article": ""
            }

        # 2. TF-IDF Vector & Out-of-Vocabulary Check
        vector = self.vectorizer.transform([cleaned_article])
        non_zero_terms = vector.nnz  # Count how many words matched the vocabulary

        # 3. Naive Bayes Probabilities
        probabilities = self.model.predict_proba(vector)[0]
        classes = list(self.model.classes_)
        real_index = classes.index("Real") if "Real" in classes else 1
        raw_real_prob = float(probabilities[real_index] * 100)

        # 4. Domain & Vocabulary Calibration
        # If the headline contains few training vocabulary terms, calibrate toward neutral-high
        if non_zero_terms < 3:
            calibrated_ml = 65.0 + (raw_real_prob * 0.25)
        else:
            calibrated_ml = raw_real_prob

        # 5. Similarity Cross-Referencing
        similarity_result = self.similarity_checker.calculate_similarity(article)
        similarity_score = float(similarity_result.get("similarity_score", 0.0))
        matched_article = similarity_result.get("matched_article", "")

        # 6. Composite Score Calculation
        # Base confidence from ML (85%) + bonus from verified corpus match (15%)
        final_score = (0.85 * calibrated_ml) + (0.15 * similarity_score)
        
        # Keep score strictly bounded between 15% and 98%
        final_score = round(max(15.0, min(98.0, final_score)), 2)
        confidence = round(calibrated_ml, 2)

        # 7. Risk Categorization
        if final_score >= 75:
            risk = "Highly Credible"
            prediction = "Real"
        elif final_score >= 50:
            risk = "Moderately Credible"
            prediction = "Likely Real"
        elif final_score >= 35:
            risk = "Low Credibility"
            prediction = "Unverified / Sensational"
        else:
            risk = "Highly Suspicious"
            prediction = "Fake"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "similarity_score": round(similarity_score, 2),
            "credibility_score": final_score,
            "credibility_level": risk,
            "matched_article": matched_article
        }


# -----------------------------------------------------
# Global Bridge for Streamlit UI Integration
# -----------------------------------------------------

_analyzer_instance = None

def get_analyzer():
    global _analyzer_instance
    if _analyzer_instance is None:
        try:
            _analyzer_instance = CredibilityAnalyzer()
        except Exception as e:
            print(f"Analyzer Init Error: {e}")
            return None
    return _analyzer_instance

def get_credibility_score(text: str) -> int:
    """Returns integer credibility percentage (0-100) for UI display."""
    analyzer = get_analyzer()
    if analyzer:
        res = analyzer.analyze(text)
        return int(round(res.get("credibility_score", 50)))
    return 50

def get_full_analysis(text: str) -> dict:
    """Returns the full metrics dictionary."""
    analyzer = get_analyzer()
    if analyzer:
        return analyzer.analyze(text)
    return {
        "prediction": "Unknown",
        "confidence": 50.0,
        "similarity_score": 0.0,
        "credibility_score": 50.0,
        "credibility_level": "Unverified",
        "matched_article": ""
    }