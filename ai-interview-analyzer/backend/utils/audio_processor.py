"""
Audio Processing Module
Handles speech-to-text transcription and audio feature extraction.
"""

import re
import math
import random
from typing import Dict, Optional


class AudioProcessor:
    """
    Processes audio files to extract speech features.

    In a production environment, this integrates with:
    - OpenAI Whisper for transcription
    - librosa for audio feature extraction

    For local/offline use, this module provides simulated features
    so the system runs without heavy GPU dependencies.
    """

    FILLER_WORDS = {
        "um", "uh", "like", "you know", "so", "basically",
        "actually", "literally", "right", "okay", "well",
        "hmm", "err", "ah", "kind of", "sort of"
    }

    def __init__(self, use_whisper: bool = False):
        self.use_whisper = use_whisper
        self._whisper_model = None

    def _load_whisper(self):
        """Lazy-load Whisper model."""
        if self._whisper_model is None:
            try:
                import whisper
                self._whisper_model = whisper.load_model("base")
            except ImportError:
                raise ImportError(
                    "openai-whisper not installed. Run: pip install openai-whisper"
                )
        return self._whisper_model

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio file to text.
        Returns transcript and word-level timestamps if available.
        """
        if self.use_whisper:
            return self._transcribe_whisper(audio_path)
        else:
            return self._transcribe_mock(audio_path)

    def _transcribe_whisper(self, audio_path: str) -> Dict:
        """Real Whisper transcription."""
        model = self._load_whisper()
        result = model.transcribe(audio_path, word_timestamps=True)
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", "en"),
            "method": "whisper"
        }

    def _transcribe_mock(self, audio_path: str) -> Dict:
        """
        Mock transcription for demo/testing without Whisper.
        Returns a realistic interview response sample.
        """
        samples = [
            "Um, so basically I worked at TechCorp for three years where I was, you know, "
            "responsible for leading a team of five engineers. Like, we built a microservices "
            "platform that reduced latency by 40%. I kind of managed the project timeline and "
            "coordinated with stakeholders. The result was, uh, a successful launch and our "
            "users were really happy with the performance improvements.",

            "I have extensive experience in software development. I led the migration of our "
            "monolithic application to a cloud-native architecture. The situation was that our "
            "system was struggling to handle peak loads. My task was to design and implement "
            "a scalable solution. I took action by proposing a phased migration approach and "
            "coordinating with cross-functional teams. The result was a 60% improvement in "
            "system reliability and a reduction in infrastructure costs.",
        ]

        # Use file path hash to select consistent sample
        idx = hash(audio_path) % len(samples)
        text = samples[idx]

        return {
            "text": text,
            "segments": [],
            "language": "en",
            "method": "mock"
        }

    def extract_audio_features(self, audio_path: str) -> Dict:
        """
        Extract acoustic features from audio file.
        Uses librosa if available, otherwise returns estimated features.
        """
        try:
            return self._extract_with_librosa(audio_path)
        except ImportError:
            return self._estimate_features(audio_path)

    def _extract_with_librosa(self, audio_path: str) -> Dict:
        """Real feature extraction using librosa."""
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Speaking rate estimation via zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        active_frames = zcr > 0.02
        speaking_ratio = active_frames.mean()

        # Pitch via fundamental frequency
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        voiced_f0 = f0[voiced_flag]
        pitch_mean = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 150.0
        pitch_std = float(np.nanstd(voiced_f0)) if len(voiced_f0) > 0 else 20.0

        # Pause detection via RMS energy
        rms = librosa.feature.rms(y=y)[0]
        silence_threshold = rms.mean() * 0.1
        silence_frames = rms < silence_threshold
        pause_count = self._count_pauses(silence_frames)

        return {
            "duration_seconds": round(duration, 2),
            "speaking_rate_estimate": round(speaking_ratio * 180, 1),  # approx WPM
            "pitch_mean_hz": round(pitch_mean, 2),
            "pitch_std_hz": round(pitch_std, 2),
            "pitch_variation": round(pitch_std / max(pitch_mean, 1) * 100, 2),
            "pause_count": pause_count,
            "pause_frequency": round(pause_count / max(duration / 60, 0.1), 2),
            "speaking_ratio": round(float(speaking_ratio), 3),
            "method": "librosa"
        }

    def _estimate_features(self, audio_path: str) -> Dict:
        """Deterministic feature estimation (no librosa needed)."""
        seed = abs(hash(audio_path)) % 1000
        rng = random.Random(seed)

        duration = rng.uniform(45, 180)
        speaking_rate = rng.uniform(110, 175)
        pitch_mean = rng.uniform(100, 220)
        pitch_std = rng.uniform(15, 45)
        pause_count = int(rng.uniform(3, 15))

        return {
            "duration_seconds": round(duration, 2),
            "speaking_rate_estimate": round(speaking_rate, 1),
            "pitch_mean_hz": round(pitch_mean, 2),
            "pitch_std_hz": round(pitch_std, 2),
            "pitch_variation": round(pitch_std / pitch_mean * 100, 2),
            "pause_count": pause_count,
            "pause_frequency": round(pause_count / (duration / 60), 2),
            "speaking_ratio": round(rng.uniform(0.70, 0.90), 3),
            "method": "estimated"
        }

    def _count_pauses(self, silence_frames, min_pause_frames: int = 10) -> int:
        """Count distinct pauses from silence frame array."""
        count = 0
        in_pause = False
        pause_len = 0
        for is_silent in silence_frames:
            if is_silent:
                pause_len += 1
                in_pause = True
            else:
                if in_pause and pause_len >= min_pause_frames:
                    count += 1
                in_pause = False
                pause_len = 0
        return count

    def compute_speaking_rate(self, transcript: str, duration_seconds: float) -> float:
        """Compute words per minute from transcript and duration."""
        if duration_seconds <= 0:
            return 0.0
        word_count = len(transcript.split())
        minutes = duration_seconds / 60
        return round(word_count / minutes, 1)
