"""
Behavioral Scoring Engine
Combines audio and NLP features to produce comprehensive interview scores.
Uses rule-based logic + lightweight ML model.
"""

import os
import json
import math
import pickle
from typing import Dict, Optional


class ScoringEngine:
    """
    Hybrid scoring engine using rule-based heuristics and ML features.

    Outputs:
    - Confidence Score (0–100)
    - Clarity Score (0–100)
    - Communication Score (0–100)
    - Nervousness Indicator (Low/Medium/High)
    - Overall Score (0–100)
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/communication_model.pkl")

    def __init__(self):
        self._model = None

    def _load_model(self):
        """Load pre-trained sklearn model if available."""
        if self._model is None and os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
            except Exception:
                pass
        return self._model

    # ─── Score Computations ──────────────────────────────────────────────────

    def compute_confidence_score(
        self,
        nlp: Dict,
        audio: Optional[Dict] = None
    ) -> int:
        """
        Confidence score based on linguistic and acoustic markers.
        """
        score = 50.0

        conf = nlp.get("confidence_indicators", {})
        conf_ratio = conf.get("confidence_ratio", 0.5)
        score += (conf_ratio - 0.5) * 40  # ±20 points

        # Penalize high filler word usage
        filler_count = nlp.get("filler_count", 0)
        word_count = max(nlp.get("word_count", 100), 1)
        filler_rate = filler_count / word_count
        score -= min(filler_rate * 200, 25)

        # Reward quantitative results
        if nlp.get("professional_tone", {}).get("has_quantitative_results"):
            score += 8

        # Reward professional vocabulary
        prof_terms = nlp.get("professional_tone", {}).get("professional_terms_count", 0)
        score += min(prof_terms * 2, 12)

        # Audio-based adjustments
        if audio:
            pitch_variation = audio.get("pitch_variation", 20)
            if pitch_variation < 5:
                score -= 8  # monotone = less confidence
            elif pitch_variation > 30:
                score -= 5  # too erratic

        return max(0, min(100, int(score)))

    def compute_clarity_score(
        self,
        nlp: Dict,
        audio: Optional[Dict] = None
    ) -> int:
        """
        Clarity score based on sentence structure and speech rate.
        """
        score = 55.0

        sent = nlp.get("sentence_stats", {})
        mean_words = sent.get("mean_words", 15)
        long_pct = sent.get("long_sentences_pct", 0)
        short_pct = sent.get("short_sentences_pct", 0)

        # Penalize overly long sentences (hard to follow)
        if mean_words > 20:
            score -= (mean_words - 20) * 1.5
        # Penalize too many short fragments
        if short_pct > 50:
            score -= (short_pct - 50) * 0.3

        # Vocabulary richness
        vocab = nlp.get("vocabulary", {})
        ttr = vocab.get("ttr", 0.5)
        score += min(ttr * 15, 15)

        # Filler words reduce clarity
        filler_count = nlp.get("filler_count", 0)
        score -= min(filler_count * 1.5, 20)

        # Audio: speaking rate
        if audio:
            rate = audio.get("speaking_rate_estimate", 140)
            if rate > 180:
                score -= (rate - 180) * 0.3  # too fast
            elif rate < 100:
                score -= (100 - rate) * 0.2  # too slow

        return max(0, min(100, int(score)))

    def compute_communication_score(
        self,
        nlp: Dict,
        audio: Optional[Dict] = None
    ) -> int:
        """
        Holistic communication score using ML if available, else rule-based.
        """
        features = self._extract_ml_features(nlp, audio)
        model = self._load_model()

        if model:
            try:
                score = float(model.predict([list(features.values())])[0])
                return max(0, min(100, int(score)))
            except Exception:
                pass

        return self._rule_based_communication(nlp, audio)

    def _rule_based_communication(
        self,
        nlp: Dict,
        audio: Optional[Dict] = None
    ) -> int:
        score = 50.0

        # STAR method presence
        star = nlp.get("star_analysis", {})
        star_score = star.get("star_score", 0)
        score += star_score * 0.2  # 0-20 points

        # Tone quality
        tone = nlp.get("professional_tone", {})
        tone_score = tone.get("tone_score", 50)
        score += (tone_score - 50) * 0.3  # ±15 points

        # Filler word penalty
        filler_count = nlp.get("filler_count", 0)
        score -= min(filler_count * 1.0, 15)

        # Sentence variety (good communication = varied structure)
        sent = nlp.get("sentence_stats", {})
        if sent.get("count", 0) >= 5:
            score += 5

        # Audio fluency
        if audio:
            pause_freq = audio.get("pause_frequency", 5)
            if pause_freq > 15:
                score -= 10  # too many pauses
            elif pause_freq < 3:
                score += 5   # very fluent

        return max(0, min(100, int(score)))

    def compute_nervousness(
        self,
        nlp: Dict,
        audio: Optional[Dict] = None
    ) -> Dict:
        """Compute nervousness level and indicator."""
        score = 0

        filler_count = nlp.get("filler_count", 0)
        word_count = max(nlp.get("word_count", 100), 1)

        # Filler rate
        filler_rate = filler_count / word_count
        if filler_rate > 0.08:
            score += 3
        elif filler_rate > 0.04:
            score += 1

        # Low confidence language
        conf = nlp.get("confidence_indicators", {})
        if conf.get("low_confidence_count", 0) > 5:
            score += 2

        # Audio markers
        if audio:
            rate = audio.get("speaking_rate_estimate", 140)
            if rate > 180:
                score += 2
            pause_freq = audio.get("pause_frequency", 5)
            if pause_freq > 15:
                score += 2

        if score >= 5:
            level = "High"
            color = "red"
        elif score >= 2:
            level = "Medium"
            color = "amber"
        else:
            level = "Low"
            color = "green"

        return {
            "level": level,
            "color": color,
            "score": score,
            "max_score": 9
        }

    def _extract_ml_features(
        self,
        nlp: Dict,
        audio: Optional[Dict] = None
    ) -> Dict:
        """Extract feature vector for ML model."""
        filler_count = nlp.get("filler_count", 0)
        word_count = max(nlp.get("word_count", 100), 1)

        features = {
            "filler_rate": filler_count / word_count,
            "filler_count": filler_count,
            "ttr": nlp.get("vocabulary", {}).get("ttr", 0.5),
            "conf_ratio": nlp.get("confidence_indicators", {}).get("confidence_ratio", 0.5),
            "star_score": nlp.get("star_analysis", {}).get("star_score", 0) / 100,
            "prof_terms": nlp.get("professional_tone", {}).get("professional_terms_count", 0),
            "has_numbers": int(nlp.get("professional_tone", {}).get("has_quantitative_results", False)),
            "mean_sentence_len": nlp.get("sentence_stats", {}).get("mean_words", 15),
            "speaking_rate": (audio or {}).get("speaking_rate_estimate", 140),
            "pause_frequency": (audio or {}).get("pause_frequency", 5),
            "pitch_variation": (audio or {}).get("pitch_variation", 20),
        }
        return features

    def generate_scores(
        self,
        nlp: Dict,
        audio: Optional[Dict] = None
    ) -> Dict:
        """Generate all scores in a single call."""
        confidence = self.compute_confidence_score(nlp, audio)
        clarity = self.compute_clarity_score(nlp, audio)
        communication = self.compute_communication_score(nlp, audio)
        nervousness = self.compute_nervousness(nlp, audio)
        overall = int((confidence + clarity + communication) / 3)

        return {
            "confidence": confidence,
            "clarity": clarity,
            "communication": communication,
            "overall": overall,
            "nervousness": nervousness,
            "grade": self._grade(overall)
        }

    def _grade(self, score: int) -> str:
        if score >= 85: return "Excellent"
        if score >= 70: return "Good"
        if score >= 55: return "Average"
        if score >= 40: return "Needs Improvement"
        return "Poor"
