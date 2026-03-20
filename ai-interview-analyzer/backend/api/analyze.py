"""
Interview Analysis Orchestrator
Coordinates audio processing, NLP, scoring, and feedback generation.
"""

from typing import Dict, Optional

from utils.audio_processor import AudioProcessor
from utils.nlp_processor import NLPProcessor
from utils.scoring_engine import ScoringEngine
from utils.feedback_generator import FeedbackGenerator


audio_proc = AudioProcessor(use_whisper=False)
nlp_proc = NLPProcessor()
scorer = ScoringEngine()
feedback_gen = FeedbackGenerator()


def analyze_interview(session_data: Dict) -> Dict:
    """
    Full pipeline:
    1. Transcribe audio (if audio_path provided)
    2. Run NLP on transcript
    3. Extract audio features
    4. Score with hybrid engine
    5. Generate actionable feedback
    """
    transcript = session_data.get("transcript", "")
    audio_path = session_data.get("audio_path", None)
    question = session_data.get("question", "Tell me about yourself.")

    # ── Step 1: Transcription ────────────────────────────────────────────────
    transcription_result = None
    if audio_path and not transcript:
        transcription_result = audio_proc.transcribe(audio_path)
        transcript = transcription_result["text"]
    elif not transcript:
        return {"error": "No transcript or audio provided"}

    # ── Step 2: Audio Feature Extraction ────────────────────────────────────
    audio_features = None
    if audio_path:
        audio_features = audio_proc.extract_audio_features(audio_path)
        # Refine speaking rate using actual transcript + duration
        if audio_features.get("duration_seconds", 0) > 0:
            computed_rate = audio_proc.compute_speaking_rate(
                transcript, audio_features["duration_seconds"]
            )
            audio_features["speaking_rate_from_transcript"] = computed_rate
            # Use transcript-based rate as primary (more accurate)
            audio_features["speaking_rate_estimate"] = computed_rate
    else:
        # Estimate from text alone (assume 2 minutes speaking time)
        word_count = len(transcript.split())
        estimated_duration = max(word_count / 140 * 60, 10)
        audio_features = {
            "duration_seconds": round(estimated_duration, 1),
            "speaking_rate_estimate": 140.0,
            "pitch_mean_hz": 150.0,
            "pitch_std_hz": 20.0,
            "pitch_variation": 13.3,
            "pause_count": 5,
            "pause_frequency": 5.0,
            "speaking_ratio": 0.80,
            "method": "text_only_estimate"
        }

    # ── Step 3: NLP Analysis ─────────────────────────────────────────────────
    nlp_result = nlp_proc.full_analysis(transcript)

    # ── Step 4: Scoring ──────────────────────────────────────────────────────
    scores = scorer.generate_scores(nlp_result, audio_features)

    # ── Step 5: Feedback Generation ──────────────────────────────────────────
    feedback = feedback_gen.generate(nlp_result, audio_features, scores, question)

    # ── Assemble Final Result ────────────────────────────────────────────────
    return {
        "transcript": transcript,
        "question": question,
        "scores": scores,
        "audio_features": audio_features,
        "nlp_analysis": {
            "filler_count": nlp_result["filler_count"],
            "filler_detail": nlp_result["filler_detail"],
            "cleaned_text": nlp_result["cleaned_text"],
            "word_count": nlp_result["word_count"],
            "sentence_stats": nlp_result["sentence_stats"],
            "vocabulary": nlp_result["vocabulary"],
            "star_analysis": nlp_result["star_analysis"],
            "confidence_indicators": nlp_result["confidence_indicators"],
            "professional_tone": nlp_result["professional_tone"],
        },
        "feedback": feedback,
        "metadata": {
            "transcription_method": (transcription_result or {}).get("method", "provided"),
            "audio_method": audio_features.get("method", "n/a"),
            "has_audio": audio_path is not None,
        }
    }
