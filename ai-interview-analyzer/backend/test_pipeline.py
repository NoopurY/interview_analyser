"""
Test Suite — AI Interview Analyzer
Validates the analysis pipeline against labeled sample transcripts.

Run from backend/ directory:
    python test_pipeline.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from api.analyze import analyze_interview

SAMPLES_PATH = "../dataset/sample_transcripts.json"

GRADE_ORDER = ["Poor", "Needs Improvement", "Average", "Good", "Excellent"]


def grade_rank(grade: str) -> int:
    for i, g in enumerate(GRADE_ORDER):
        if g in grade:
            return i
    return -1


def run_tests():
    with open(SAMPLES_PATH) as f:
        samples = json.load(f)

    print("=" * 60)
    print("  AI Interview Analyzer — Pipeline Test Suite")
    print("=" * 60)

    passed = 0
    results = []

    for s in samples:
        result = analyze_interview({
            "transcript": s["transcript"],
            "question": s["question"],
        })

        scores = result["scores"]
        actual_grade = scores["grade"]
        expected_grade = s["expected_grade"]

        # Allow ±1 grade tolerance
        diff = abs(grade_rank(actual_grade) - grade_rank(expected_grade))
        ok = diff <= 1

        if ok:
            passed += 1

        results.append({
            "id": s["id"],
            "label": s["label"],
            "expected": expected_grade,
            "actual": actual_grade,
            "scores": scores,
            "ok": ok,
        })

        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"\n{status}  [{s['id']}] {s['label'].upper()}")
        print(f"       Expected: {expected_grade:18s}  Got: {actual_grade}")
        print(f"       Conf={scores['confidence']}  Clar={scores['clarity']}  Comm={scores['communication']}  Overall={scores['overall']}")
        print(f"       Fillers: {result['nlp_analysis']['filler_count']}  STAR: {result['nlp_analysis']['star_analysis']['star_score']}/100  Nervousness: {scores['nervousness']['level']}")
        print(f"       Notes: {s.get('notes','')}")

    print()
    print("=" * 60)
    print(f"  Results: {passed}/{len(samples)} passed")
    print("=" * 60)

    # Show scoring spread
    print("\n📊 Score Spread:")
    print(f"  {'Sample':<12} {'Conf':>6} {'Clar':>6} {'Comm':>6} {'Overall':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for r in results:
        s = r["scores"]
        print(f"  {r['label']:<12} {s['confidence']:>6} {s['clarity']:>6} {s['communication']:>6} {s['overall']:>8}  ({s['grade']})")

    return passed == len(samples)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
