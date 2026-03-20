"""
NLP Processing Module
Analyzes transcript text for communication quality metrics.
"""

import re
import math
from collections import Counter
from typing import Dict, List, Tuple


class NLPProcessor:
    """
    Natural Language Processing for interview transcript analysis.

    Computes:
    - Filler word frequency
    - Vocabulary richness (TTR, hapax legomena ratio)
    - Sentence structure metrics
    - Confidence / clarity indicators
    - STAR method detection
    """

    FILLER_WORDS = [
        "um", "uh", "like", "you know", "so", "basically",
        "actually", "literally", "right", "okay", "well",
        "hmm", "err", "ah", "kind of", "sort of", "i mean",
        "to be honest", "honestly", "you see", "anyway"
    ]

    STAR_KEYWORDS = {
        "situation": [
            "situation", "context", "background", "at the time",
            "was working", "were facing", "project was", "team was",
            "company was", "when i was", "in my role", "when i joined",
            "our system", "our team", "our company", "the company",
            "we were", "we had", "facing", "experiencing", "struggling",
            "at my previous", "at my last", "while working"
        ],
        "task": [
            "task", "responsibility", "my job", "i was responsible",
            "i needed to", "my goal was", "objective", "challenge was",
            "problem was", "required to", "had to", "my role was",
            "my responsibility", "tasked with", "assigned to",
            "my mission", "goal was to", "aim was"
        ],
        "action": [
            "i did", "i took", "i implemented", "i designed",
            "i built", "i created", "i led", "i coordinated",
            "i proposed", "i developed", "i worked", "i decided",
            "i initiated", "i managed", "i resolved", "i assembled",
            "i established", "i introduced", "i restructured",
            "i deployed", "i migrated", "i mentored", "i presented",
            "i drove", "i launched", "i trained", "i personally",
            "i owned", "i spearheaded", "i redesigned", "i refactored"
        ],
        "result": [
            "result", "outcome", "achieved", "improved", "increased",
            "decreased", "reduced", "saved", "generated", "delivered",
            "completed", "successfully", "as a result", "consequently",
            "percent", "%", "times faster", "revenue", "recovered",
            "exceeded", "surpassed", "dropped", "eliminated", "grew",
            "cut", "boosted", "million", "billion", "thousand",
            "the result", "this resulted", "ultimately", "on time",
            "ahead of schedule", "launched"
        ]
    }

    CONFIDENCE_INDICATORS = {
        "high": [
            "i am", "i have", "i will", "i can", "i did",
            "i led", "i built", "i achieved", "i delivered",
            "absolutely", "definitely", "certainly", "confident",
            "successfully", "accomplished", "proven",
            "i implemented", "i designed", "i created", "i managed",
            "i drove", "i launched", "i established", "i deployed",
            "i assembled", "i mentored", "i presented", "i owned",
            "i spearheaded", "i coordinated", "i resolved",
            "i developed", "i redesigned", "i personally"
        ],
        "low": [
            "i think", "i guess", "maybe", "perhaps", "sort of",
            "kind of", "i'm not sure", "i hope", "possibly",
            "i believe", "might", "i'm trying",
            "attempted", "tried to", "not really", "i suppose"
        ]
    }

    PROFESSIONAL_TERMS = [
        "stakeholder", "deliverable", "kpi", "roi", "agile",
        "scrum", "sprint", "deployment", "scalability", "architecture",
        "optimization", "collaboration", "initiative", "strategy",
        "impact", "metrics", "performance", "leadership", "framework",
        "implementation", "infrastructure", "methodology", "milestone"
    ]

    def __init__(self):
        pass

    def clean_transcript(self, text: str) -> Tuple[str, int]:
        """
        Remove filler words from transcript.
        Returns (cleaned_text, filler_count).
        """
        text_lower = text.lower()
        filler_count = 0
        cleaned = text

        # Count multi-word fillers first
        for filler in sorted(self.FILLER_WORDS, key=len, reverse=True):
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, text_lower)
            filler_count += len(matches)
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Clean extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned, filler_count

    def count_fillers(self, text: str) -> Dict[str, int]:
        """Return per-filler-word counts."""
        text_lower = text.lower()
        counts = {}
        for filler in self.FILLER_WORDS:
            pattern = r'\b' + re.escape(filler) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                counts[filler] = count
        return counts

    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def sentence_length_distribution(self, text: str) -> Dict:
        """Analyze sentence length statistics."""
        sentences = self.extract_sentences(text)
        if not sentences:
            return {"mean": 0, "min": 0, "max": 0, "short": 0, "long": 0}

        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        short = sum(1 for l in lengths if l < 8)
        long = sum(1 for l in lengths if l > 25)

        return {
            "count": len(sentences),
            "mean_words": round(mean_len, 1),
            "min_words": min(lengths),
            "max_words": max(lengths),
            "short_sentences_pct": round(short / len(lengths) * 100, 1),
            "long_sentences_pct": round(long / len(lengths) * 100, 1),
        }

    def vocabulary_richness(self, text: str) -> Dict:
        """
        Compute vocabulary diversity metrics.
        - Type-Token Ratio (TTR): unique/total words
        - Hapax Legomena: words appearing only once
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if not words:
            return {"ttr": 0, "hapax_ratio": 0, "unique_words": 0, "total_words": 0}

        word_freq = Counter(words)
        unique = len(word_freq)
        total = len(words)
        hapax = sum(1 for w, c in word_freq.items() if c == 1 and len(w) > 3)

        # Corrected TTR (handles longer texts fairly)
        ttr = unique / math.sqrt(2 * total) if total > 0 else 0

        return {
            "ttr": round(ttr, 3),
            "type_token_ratio": round(unique / total, 3) if total > 0 else 0,
            "hapax_ratio": round(hapax / unique, 3) if unique > 0 else 0,
            "unique_words": unique,
            "total_words": total,
            "professional_terms": self._count_professional_terms(text)
        }

    def _count_professional_terms(self, text: str) -> int:
        """Count professional/technical vocabulary usage."""
        text_lower = text.lower()
        return sum(1 for term in self.PROFESSIONAL_TERMS if term in text_lower)

    def detect_star_method(self, text: str) -> Dict:
        """
        Detect STAR method components in interview answer.
        Returns presence scores and missing components.
        """
        text_lower = text.lower()
        component_scores = {}

        for component, keywords in self.STAR_KEYWORDS.items():
            matches = []
            for kw in keywords:
                if kw in text_lower:
                    matches.append(kw)
            score = min(len(matches) / 2, 1.0)  # normalize to 0-1
            component_scores[component] = {
                "detected": score > 0.3,
                "confidence": round(score, 2),
                "matched_keywords": matches[:3]  # top 3 matches
            }

        detected = [k for k, v in component_scores.items() if v["detected"]]
        missing = [k for k, v in component_scores.items() if not v["detected"]]

        star_score = len(detected) / 4 * 100

        return {
            "star_score": round(star_score),
            "components": component_scores,
            "detected_components": detected,
            "missing_components": missing,
            "has_full_star": len(detected) == 4
        }

    def analyze_confidence_indicators(self, text: str) -> Dict:
        """
        Analyze linguistic markers of confidence vs uncertainty.
        """
        text_lower = text.lower()
        words = text_lower.split()
        total = max(len(words), 1)

        high_conf = sum(
            len(re.findall(r'\b' + re.escape(ind) + r'\b', text_lower))
            for ind in self.CONFIDENCE_INDICATORS["high"]
        )
        low_conf = sum(
            len(re.findall(r'\b' + re.escape(ind) + r'\b', text_lower))
            for ind in self.CONFIDENCE_INDICATORS["low"]
        )

        # Normalize per 100 words
        high_rate = (high_conf / total) * 100
        low_rate = (low_conf / total) * 100

        confidence_ratio = high_conf / max(high_conf + low_conf, 1)

        return {
            "high_confidence_count": high_conf,
            "low_confidence_count": low_conf,
            "high_confidence_rate": round(high_rate, 2),
            "low_confidence_rate": round(low_rate, 2),
            "confidence_ratio": round(confidence_ratio, 3),
            "nervousness_indicators": low_conf
        }

    def analyze_professional_tone(self, text: str) -> Dict:
        """Evaluate professional tone and structure."""
        prof_terms = self._count_professional_terms(text)
        sentences = self.extract_sentences(text)
        sent_count = max(len(sentences), 1)

        # Quantitative results mentioned
        has_numbers = bool(re.search(r'\d+%|\d+x|\$\d+|\d+\s*(percent|times|million|billion)', text.lower()))

        # First-person accountability
        first_person = len(re.findall(r'\bi\b', text.lower()))
        first_person_rate = round(first_person / sent_count, 2)

        # Passive voice detection (simple heuristic)
        passive_patterns = re.findall(r'\b(was|were|been|being)\s+\w+ed\b', text.lower())
        passive_rate = round(len(passive_patterns) / sent_count, 2)

        return {
            "professional_terms_count": prof_terms,
            "has_quantitative_results": has_numbers,
            "first_person_rate": first_person_rate,
            "passive_voice_count": len(passive_patterns),
            "passive_voice_rate": passive_rate,
            "tone_score": self._compute_tone_score(prof_terms, has_numbers, passive_rate)
        }

    def _compute_tone_score(self, prof_terms: int, has_numbers: bool, passive_rate: float) -> int:
        score = 50
        score += min(prof_terms * 5, 25)
        if has_numbers:
            score += 15
        score -= min(passive_rate * 10, 20)
        return max(0, min(100, int(score)))

    def full_analysis(self, text: str) -> Dict:
        """Run all NLP analyses and return combined result."""
        cleaned, filler_count = self.clean_transcript(text)
        filler_detail = self.count_fillers(text)

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "filler_count": filler_count,
            "filler_detail": filler_detail,
            "sentence_stats": self.sentence_length_distribution(text),
            "vocabulary": self.vocabulary_richness(cleaned),
            "star_analysis": self.detect_star_method(text),
            "confidence_indicators": self.analyze_confidence_indicators(text),
            "professional_tone": self.analyze_professional_tone(text),
            "word_count": len(text.split()),
            "character_count": len(text)
        }
