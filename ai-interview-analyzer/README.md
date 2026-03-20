# 🎙️ AI Interview Analyzer
### Multimodal Communication & Behavioral Intelligence System

A production-ready AI system that analyzes interview responses through NLP and audio processing, generating deep behavioral feedback with scores, STAR method detection, and actionable improvement plans.

---

## 🚀 Quick Start

### Option A — Frontend Only (No Setup Required)
Just open `frontend/index.html` in your browser. The system runs fully client-side.
Click **"Try Demo"** to load a sample and analyze immediately.

### Option B — Full Stack (with Python Backend)

```bash
# 1. Clone / enter project
cd ai-interview-analyzer/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Train ML model
cd models && python train_model.py && cd ..

# 5. Start backend
python app.py
# → Running on http://localhost:5000

# 6. Open frontend
open ../frontend/index.html
```

---

## 📁 Project Structure

```
ai-interview-analyzer/
├── frontend/
│   └── index.html              # Full UI (self-contained)
│
├── backend/
│   ├── app.py                  # Flask API server
│   ├── requirements.txt        # Python dependencies
│   ├── api/
│   │   └── analyze.py          # Analysis orchestrator
│   ├── utils/
│   │   ├── audio_processor.py  # Speech-to-text + audio features
│   │   ├── nlp_processor.py    # Text analysis (NLP)
│   │   ├── scoring_engine.py   # Hybrid scoring (rules + ML)
│   │   └── feedback_generator.py # Smart feedback generation
│   └── models/
│       └── train_model.py      # ML model training script
│
└── README.md
```

---

## 🧠 Core Features

### 1. Audio Processing
- Accepts WAV, MP3, M4A, OGG files
- Integrates with **OpenAI Whisper** for transcription (optional)
- Extracts acoustic features via **librosa** (optional):
  - Speaking rate (WPM)
  - Pause frequency
  - Pitch mean & variation

### 2. NLP Analysis
- Filler word detection & count (um, uh, like, you know…)
- Sentence length distribution
- Vocabulary richness (Type-Token Ratio, hapax legomena)
- Professional terminology scoring
- Passive voice detection
- Quantified results detection

### 3. Behavioral Scoring Engine
| Score | What It Measures |
|-------|-----------------|
| **Confidence** (0–100) | Assertive language, ownership, no hedging |
| **Clarity** (0–100) | Filler reduction, sentence structure, vocabulary |
| **Communication** (0–100) | STAR coverage, tone, quantified results |
| **Nervousness** | Low / Medium / High indicator |
| **Overall** (0–100) | Composite performance grade |

### 4. STAR Method Detector
Detects all 4 behavioral interview components:
- **S**ituation — context/background
- **T**ask — responsibility/goal
- **A**ction — what YOU did
- **R**esult — measurable outcome

### 5. Smart Feedback Generator
Produces 8–12 detailed feedback items per analysis:
- Categorized: Strengths, Improvements, Warnings
- Each item has title, explanation, and actionable tip
- Priority list of top 3 improvements

---

## 🖥️ UI Pages

### Landing Page
- Dark glassmorphism design with animated background orbs
- File upload zone (drag & drop)
- Transcript text input
- Demo mode for instant testing

### Analysis Dashboard
- 4 animated circular score indicators (SVG)
- Nervousness level with progress bar
- 8-panel metrics grid
- Bar chart: filler word breakdown
- Radar chart: all 4 score dimensions
- STAR method 4-card visualization
- Tabbed feedback panel (All / Strengths / Improvements / Warnings)
- Transcript viewer

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/health` | Health check |
| POST | `/api/upload` | Upload audio file |
| POST | `/api/analyze` | Trigger analysis |
| GET  | `/api/results/<id>` | Get results |
| POST | `/api/analyze-text` | Direct text analysis |

### Example API Call

```bash
curl -X POST http://localhost:5000/api/analyze-text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Um, so basically I led a team of 5 engineers...",
    "question": "Tell me about a time you led a project."
  }'
```

---

## 📦 Optional Dependencies

For full audio analysis:

```bash
# Install Whisper (speech-to-text)
pip install openai-whisper

# Install librosa (audio features)
pip install librosa soundfile

# Install PyTorch (required by Whisper)
pip install torch
```

Enable in `audio_processor.py`:
```python
audio_proc = AudioProcessor(use_whisper=True)
```

---

## 🧪 ML Model

The scoring engine uses a **RandomForest regressor** trained on synthetic data:

```bash
cd backend/models
python train_model.py
```

**Features used:**
- Filler rate, filler count
- Type-Token Ratio
- Confidence ratio
- STAR score
- Professional terms count
- Quantified results (binary)
- Mean sentence length
- Speaking rate
- Pause frequency
- Pitch variation

**Evaluation (typical):**
- MAE: ~4–6 points
- R²: ~0.88–0.92

---

## 🎯 Sample Output

```json
{
  "scores": {
    "confidence": 62,
    "clarity": 54,
    "communication": 67,
    "overall": 61,
    "grade": "Average",
    "nervousness": { "level": "Medium" }
  },
  "nlp_analysis": {
    "filler_count": 9,
    "star_analysis": {
      "star_score": 75,
      "detected_components": ["situation", "task", "action"],
      "missing_components": ["result"]
    }
  },
  "feedback": {
    "summary": "Good response overall...",
    "top_priority": [
      { "title": "Add quantified results", "tip": "Add numbers: '40% faster', '$80K saved'" },
      { "title": "High filler frequency (9 instances)", "tip": "Use pause-and-breathe technique" }
    ]
  }
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3 (glassmorphism), Vanilla JS |
| Charts | Chart.js 4.x |
| Backend | Python 3.10+, Flask 3.0 |
| ML | scikit-learn (RandomForest) |
| Audio | OpenAI Whisper (optional), librosa (optional) |
| Fonts | Syne (headings), DM Sans (body) |

---

## 📸 Visual Design

- **Theme**: Dark glassmorphism with gradient orbs
- **Colors**: Electric blue (#63caff), violet (#a78bfa), emerald (#34d399)
- **Typography**: Syne 800 for headings, DM Sans for body
- **Animations**: Smooth SVG circle fills, counter animations, staggered reveals
- **Charts**: Transparent bar + radar charts with custom color palettes

---

## 🔒 Privacy

All analysis runs locally. No audio or text is sent to external servers (unless you use the hosted Whisper API separately).

---

## 📄 License

MIT License — free to use, modify, and distribute.
