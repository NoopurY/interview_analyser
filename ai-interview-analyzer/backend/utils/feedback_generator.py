"""
Smart Feedback Generator
Produces detailed, actionable feedback from analysis data.
"""

from typing import Dict, List


class FeedbackGenerator:
    """
    Generates human-readable, actionable interview feedback.
    """

    def generate(
        self,
        nlp: Dict,
        audio: Dict,
        scores: Dict,
        question: str = ""
    ) -> Dict:
        """
        Main entry point. Returns structured feedback object.
        """
        items = []
        items.extend(self._filler_feedback(nlp))
        items.extend(self._speaking_rate_feedback(audio))
        items.extend(self._star_feedback(nlp))
        items.extend(self._vocabulary_feedback(nlp))
        items.extend(self._confidence_feedback(nlp, scores))
        items.extend(self._sentence_feedback(nlp))
        items.extend(self._tone_feedback(nlp))
        items.extend(self._nervousness_feedback(scores))

        strengths = [i for i in items if i["type"] == "strength"]
        improvements = [i for i in items if i["type"] == "improvement"]
        warnings = [i for i in items if i["type"] == "warning"]

        summary = self._generate_summary(scores, nlp, len(improvements))

        return {
            "summary": summary,
            "strengths": strengths,
            "improvements": improvements,
            "warnings": warnings,
            "all_feedback": items,
            "top_priority": improvements[:3] if improvements else [],
        }

    # ─── Filler Word Feedback ────────────────────────────────────────────────

    def _filler_feedback(self, nlp: Dict) -> List[Dict]:
        items = []
        filler_count = nlp.get("filler_count", 0)
        word_count = max(nlp.get("word_count", 100), 1)
        filler_rate = filler_count / word_count * 100
        filler_detail = nlp.get("filler_detail", {})

        if filler_count == 0:
            items.append({
                "type": "strength",
                "category": "Clarity",
                "icon": "✅",
                "title": "Zero filler words",
                "detail": "Excellent! You used no filler words, demonstrating strong verbal control and confidence."
            })
        elif filler_count <= 3:
            items.append({
                "type": "strength",
                "category": "Clarity",
                "icon": "👍",
                "title": f"Minimal filler usage ({filler_count} instances)",
                "detail": "Very few filler words detected. This shows good preparation and communication control."
            })
        elif filler_count <= 8:
            top = sorted(filler_detail.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f'"{w}" ({c}x)' for w, c in top)
            items.append({
                "type": "improvement",
                "category": "Clarity",
                "icon": "⚠️",
                "title": f"Moderate filler word usage ({filler_count} instances, {filler_rate:.1f}% of speech)",
                "detail": f"Your most used fillers were: {top_str}. Practice pausing silently instead of filling gaps.",
                "tip": "Record yourself answering a question and count fillers. Aim for fewer than 5 per minute."
            })
        else:
            top = sorted(filler_detail.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f'"{w}" ({c}x)' for w, c in top)
            items.append({
                "type": "warning",
                "category": "Clarity",
                "icon": "🚨",
                "title": f"High filler word frequency ({filler_count} instances)",
                "detail": f"You used filler words {filler_count} times, which significantly reduces perceived confidence. Most frequent: {top_str}.",
                "tip": "Practice the 'pause-and-breathe' technique: when you feel a filler word coming, pause for 1 second instead."
            })

        return items

    # ─── Speaking Rate Feedback ───────────────────────────────────────────────

    def _speaking_rate_feedback(self, audio: Dict) -> List[Dict]:
        items = []
        if not audio:
            return items

        rate = audio.get("speaking_rate_estimate", 140)

        if 120 <= rate <= 160:
            items.append({
                "type": "strength",
                "category": "Delivery",
                "icon": "🎙️",
                "title": f"Optimal speaking pace ({rate:.0f} WPM)",
                "detail": "Your speaking rate is in the ideal range (120–160 WPM), making it easy for the listener to follow along."
            })
        elif rate > 180:
            items.append({
                "type": "warning",
                "category": "Delivery",
                "icon": "⚡",
                "title": f"Speaking too fast ({rate:.0f} WPM)",
                "detail": f"At {rate:.0f} words per minute, you may be hard to follow. Ideal range is 120–160 WPM.",
                "tip": "Practice with a metronome app to slow down. Stress key words and take deliberate pauses."
            })
        elif rate > 160:
            items.append({
                "type": "improvement",
                "category": "Delivery",
                "icon": "🏃",
                "title": f"Slightly fast pace ({rate:.0f} WPM)",
                "detail": "Your pace is slightly above the ideal range. Slow down during key points to emphasize them.",
                "tip": "Use pauses after important statements for emphasis."
            })
        elif rate < 100:
            items.append({
                "type": "improvement",
                "category": "Delivery",
                "icon": "🐢",
                "title": f"Speaking pace too slow ({rate:.0f} WPM)",
                "detail": "Speaking too slowly can make you seem hesitant. Aim for 120–160 WPM for a natural, confident delivery.",
                "tip": "Practice with a timer. Aim to cover key points more concisely."
            })

        # Pitch variation feedback
        pitch_variation = audio.get("pitch_variation", 20)
        if pitch_variation < 8:
            items.append({
                "type": "improvement",
                "category": "Delivery",
                "icon": "📊",
                "title": "Monotone delivery detected",
                "detail": "Low pitch variation makes speech less engaging. Vary your tone to emphasize key points.",
                "tip": "Practice reading passages aloud with exaggerated intonation first, then moderate it."
            })
        elif pitch_variation > 15:
            items.append({
                "type": "strength",
                "category": "Delivery",
                "icon": "🎵",
                "title": "Good vocal variety",
                "detail": "You demonstrated good pitch variation, keeping your delivery engaging and dynamic."
            })

        return items

    # ─── STAR Method Feedback ────────────────────────────────────────────────

    def _star_feedback(self, nlp: Dict) -> List[Dict]:
        items = []
        star = nlp.get("star_analysis", {})
        missing = star.get("missing_components", [])
        detected = star.get("detected_components", [])
        star_score = star.get("star_score", 0)

        if star.get("has_full_star"):
            items.append({
                "type": "strength",
                "category": "Structure",
                "icon": "⭐",
                "title": "Complete STAR method used",
                "detail": "Excellent! Your response includes all four STAR components: Situation, Task, Action, and Result. This is the gold standard for behavioral interviews."
            })
        elif len(detected) >= 3:
            missing_str = ", ".join(c.capitalize() for c in missing)
            items.append({
                "type": "improvement",
                "category": "Structure",
                "icon": "⭐",
                "title": f"STAR method mostly present (missing: {missing_str})",
                "detail": f"You used {len(detected)}/4 STAR components. Adding the {missing_str} component(s) would significantly strengthen your answer.",
                "tip": f"For '{missing[0].capitalize()}': " + self._star_tip(missing[0]) if missing else ""
            })
        elif len(detected) >= 2:
            missing_str = ", ".join(c.capitalize() for c in missing)
            items.append({
                "type": "warning",
                "category": "Structure",
                "icon": "📋",
                "title": f"STAR method partially used ({len(detected)}/4 components)",
                "detail": f"Your answer is missing: {missing_str}. Structured answers score much higher with interviewers.",
                "tip": "Template: 'In [situation], my task was to [task]. I did [actions]. The result was [measurable result].'"
            })
        else:
            items.append({
                "type": "warning",
                "category": "Structure",
                "icon": "🔴",
                "title": "STAR method not detected",
                "detail": "Your answer lacks structured storytelling. Interviewers use STAR to evaluate behavioral competencies.",
                "tip": "Before answering: jot down S-T-A-R bullet points. Mention numbers in the Result (e.g., '40% faster', '$50K saved')."
            })

        return items

    def _star_tip(self, component: str) -> str:
        tips = {
            "situation": "Briefly describe the context (team size, project, timeframe).",
            "task": "State what your specific responsibility or goal was.",
            "action": "Describe the exact steps YOU took (use 'I', not 'we').",
            "result": "Quantify the outcome: %, $, time saved, user count, etc."
        }
        return tips.get(component, "")

    # ─── Vocabulary Feedback ─────────────────────────────────────────────────

    def _vocabulary_feedback(self, nlp: Dict) -> List[Dict]:
        items = []
        vocab = nlp.get("vocabulary", {})
        ttr = vocab.get("ttr", 0.5)
        prof_terms = vocab.get("professional_terms", 0)

        if ttr >= 0.7:
            items.append({
                "type": "strength",
                "category": "Vocabulary",
                "icon": "📚",
                "title": "Rich vocabulary",
                "detail": f"Your vocabulary diversity score is {ttr:.2f} (excellent). Varied word choice keeps answers engaging."
            })
        elif ttr < 0.4:
            items.append({
                "type": "improvement",
                "category": "Vocabulary",
                "icon": "📖",
                "title": "Limited vocabulary diversity",
                "detail": "You repeated many words, reducing the sophistication of your answer.",
                "tip": "Expand your industry vocabulary. Prepare 10–15 domain-specific terms to use naturally in answers."
            })

        if prof_terms >= 3:
            items.append({
                "type": "strength",
                "category": "Vocabulary",
                "icon": "💼",
                "title": f"Strong professional vocabulary ({prof_terms} industry terms)",
                "detail": "You used relevant professional terminology, signaling domain expertise."
            })
        elif prof_terms == 0:
            items.append({
                "type": "improvement",
                "category": "Vocabulary",
                "icon": "🏢",
                "title": "No professional terminology detected",
                "detail": "Adding domain-specific vocabulary (e.g., 'stakeholder alignment', 'KPIs', 'scalability') signals expertise.",
                "tip": "Research 5 core terms in your field and practice weaving them naturally into answers."
            })

        return items

    # ─── Confidence Feedback ─────────────────────────────────────────────────

    def _confidence_feedback(self, nlp: Dict, scores: Dict) -> List[Dict]:
        items = []
        conf_score = scores.get("confidence", 50)
        conf_indicators = nlp.get("confidence_indicators", {})
        low_count = conf_indicators.get("low_confidence_count", 0)

        if conf_score >= 75:
            items.append({
                "type": "strength",
                "category": "Confidence",
                "icon": "💪",
                "title": f"Strong confidence signals (score: {conf_score}/100)",
                "detail": "Your language demonstrates ownership and assertiveness. Interviewers respond positively to confident delivery."
            })
        elif conf_score < 50:
            items.append({
                "type": "improvement",
                "category": "Confidence",
                "icon": "🎯",
                "title": f"Low confidence language detected (score: {conf_score}/100)",
                "detail": f"You used {low_count} hedging phrases ('I think', 'maybe', 'I guess'). These undermine your credibility.",
                "tip": "Replace 'I think I can handle that' with 'I have handled that before — here's how.'"
            })

        return items

    # ─── Sentence Structure Feedback ─────────────────────────────────────────

    def _sentence_feedback(self, nlp: Dict) -> List[Dict]:
        items = []
        sent = nlp.get("sentence_stats", {})
        mean_len = sent.get("mean_words", 15)
        long_pct = sent.get("long_sentences_pct", 0)
        count = sent.get("count", 0)

        if long_pct > 40:
            items.append({
                "type": "improvement",
                "category": "Clarity",
                "icon": "✂️",
                "title": f"{long_pct:.0f}% of sentences are overly long",
                "detail": "Long, complex sentences are hard to follow in spoken context. Break them into shorter, punchy statements.",
                "tip": "After each sentence, ask: 'Can this be two sentences?' If yes, split it."
            })
        elif count >= 5 and mean_len <= 18:
            items.append({
                "type": "strength",
                "category": "Clarity",
                "icon": "📝",
                "title": "Well-structured sentences",
                "detail": f"Average sentence length of {mean_len:.0f} words is ideal for spoken communication."
            })

        return items

    # ─── Tone Feedback ───────────────────────────────────────────────────────

    def _tone_feedback(self, nlp: Dict) -> List[Dict]:
        items = []
        tone = nlp.get("professional_tone", {})
        has_numbers = tone.get("has_quantitative_results", False)
        passive_rate = tone.get("passive_voice_rate", 0)

        if has_numbers:
            items.append({
                "type": "strength",
                "category": "Impact",
                "icon": "📈",
                "title": "Quantitative results included",
                "detail": "You backed claims with numbers. Quantified achievements are far more compelling to interviewers."
            })
        else:
            items.append({
                "type": "improvement",
                "category": "Impact",
                "icon": "📉",
                "title": "No quantified results detected",
                "detail": "Vague claims like 'improved performance' are much weaker than '40% latency reduction'.",
                "tip": "For every achievement, ask: 'By how much? How many? Over what timeframe? What $value?'"
            })

        if passive_rate > 0.5:
            items.append({
                "type": "improvement",
                "category": "Impact",
                "icon": "✏️",
                "title": "High passive voice usage",
                "detail": "Passive constructions ('was implemented', 'was built') reduce your perceived ownership.",
                "tip": "Replace passive voice: 'The system was improved' → 'I improved the system by redesigning the cache layer.'"
            })

        return items

    # ─── Nervousness Feedback ────────────────────────────────────────────────

    def _nervousness_feedback(self, scores: Dict) -> List[Dict]:
        items = []
        nerv = scores.get("nervousness", {})
        level = nerv.get("level", "Low")

        if level == "High":
            items.append({
                "type": "warning",
                "category": "Composure",
                "icon": "😰",
                "title": "High nervousness indicators detected",
                "detail": "Multiple markers suggest significant interview anxiety (fast speech, filler words, hedging language).",
                "tip": "Before the interview: use box breathing (4-4-4-4). During: speak 20% slower than feels natural."
            })
        elif level == "Medium":
            items.append({
                "type": "improvement",
                "category": "Composure",
                "icon": "😐",
                "title": "Moderate nervousness detected",
                "detail": "Some anxiety markers present. With practice, you can significantly reduce these signals.",
                "tip": "Do mock interviews 3–5 times before the real one. Familiarity drastically reduces anxiety."
            })
        else:
            items.append({
                "type": "strength",
                "category": "Composure",
                "icon": "😎",
                "title": "Calm and composed delivery",
                "detail": "You showed minimal nervousness indicators. Your composure will inspire confidence in interviewers."
            })

        return items

    # ─── Summary ─────────────────────────────────────────────────────────────

    def _generate_summary(self, scores: Dict, nlp: Dict, improvement_count: int) -> str:
        overall = scores.get("overall", 50)
        grade = scores.get("grade", "Average")
        conf = scores.get("confidence", 50)
        clarity = scores.get("clarity", 50)

        if overall >= 80:
            opening = "Outstanding performance! Your interview response demonstrates strong communication skills and professional presence."
        elif overall >= 65:
            opening = "Good response overall. You showed solid communication skills with some areas to refine."
        elif overall >= 50:
            opening = "Decent attempt. Your response shows potential, but targeted improvements will significantly boost your impact."
        else:
            opening = "Your response needs work in several areas. The good news: all these skills are learnable with deliberate practice."

        filler_count = nlp.get("filler_count", 0)
        star = nlp.get("star_analysis", {})
        star_comps = len(star.get("detected_components", []))

        detail = f" Confidence is rated {conf}/100, clarity at {clarity}/100."
        if filler_count > 5:
            detail += f" Reducing your {filler_count} filler words would immediately boost your score."
        if star_comps < 4:
            detail += f" Adding missing STAR components ({', '.join(c.capitalize() for c in star.get('missing_components', []))}) would strengthen structure."

        return opening + detail
