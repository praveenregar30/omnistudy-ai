# Devpost Submission Text

## Project Title
**OmniStudy AI: Source-Grounded Multimodal Study Companion & Adaptive Learner Engine**

## Tagline
An intelligent study companion that unifies lecture videos, textbooks, and slides into a source-cited knowledge base with Bayesian Knowledge Tracing, multi-format adaptive assessments, and isolated learner profiles.

---

## 💡 Inspiration
Modern university students learn across disjointed mediums: 2-hour lecture video recordings, 800-page textbook PDFs, and 60-slide PowerPoint presentations. When students turn to standard LLM chatbots, two critical failure modes emerge:
1. **Hallucination & Lack of Syllabus Grounding:** Generic chatbots give explanations that do not match the professor's notation, textbook definitions, or exam scope, and cannot point the student to the exact minute in the lecture video where the concept was taught.
2. **Amnesia & Lack of Cognitive Modeling:** General chatbots treat every interaction as isolated, with zero understanding of what the student already knows, what they struggle with, or when they are about to forget a concept.
3. **Data Leakage & Public Contamination:** When multiple students or teachers share an educational platform, notes, scores, and chat logs are frequently leaked across sessions.

We built **OmniStudy AI** for the **Multimodal AI Hackathon 2026 (Track D: Personalized Tutoring & Adaptive Learning)** to solve this problem by uniting multimodal source grounding with cognitive psychology, Bayesian Knowledge Tracing, and isolated student persistence.

---

## 🚀 What It Does

OmniStudy AI transforms raw university course assets and uploaded notes into an interactive, personalized learning environment:

1. **Multimodal Knowledge Ingestion (Tab 1):** Ingests lecture video transcripts with second-level timestamps (`[MM:SS]`), paginated textbook chapters, slide decks with diagram/figure captions, and student-uploaded PDFs. Automatically chunks content, extracts formulas, and infers prerequisite DAG relationships.
2. **Source-Grounding & Exact Location Deep Links (Tab 2):** When explaining any concept, OmniStudy AI cites the exact lecture video timestamp, textbook page, and slide number, with interactive jump actions, verbatim passage quotes, and diagram captions.
3. **Curriculum Guard & Segregated Outside Knowledge Mode (Tab 2):** Queries outside the course syllabus are strictly refused with a 100% accuracy rate by default. When the user enables "Allow Outside Knowledge Mode", general external knowledge is provided under an unmistakable `[⚠️ NOT FROM YOUR COURSE MATERIAL — GENERAL AI KNOWLEDGE]` warning banner.
4. **Adaptive Assessment Engine Across 3 Question Formats (Tab 3):**
   - **MCQs** with distractor misconception analysis and hints.
   - **Short-Answer Questions** evaluated against source passage rubrics with partial credit scoring.
   - **Numerical / Computational Problems** computed and verified with Python mathematical tolerances ($\pm 0.01$).
   - **Timed Mock Exam Mode** featuring a 5-question timed test paper, timer countdown, and an automated **Post-Assessment Diagnostic Report** mapping learning gaps to timestamped review locations.
   - **Feynman Teach-Back Challenge** evaluating conceptual clarity when students explain concepts in their own words.
5. **48-Hour Emergency Exam Cram Mode & Weak-Topic Deep Remediation (Tab 4):**
   - Built for students facing an exam in 2 days who haven't studied yet: select a course topic or paste/upload notes to launch an intensive crash course from scratch with spoken audio briefs and hour-by-hour 48-hour survival schedule.
   - Runs a 4-question Diagnostic Blitz Quiz that automatically separates **Strong Topics 🟢** (exam-ready) from **Critical Weak Topics 🔴** (exam hazards).
   - **Targeted Deep Remediation:** AI re-teaches weak topics deeply with misconception breakdowns, exam tips, and follow-up concept checks that update BKT mastery in real-time upon mastery.
6. **Bayesian Knowledge Tracing (BKT) & Learner Modeling (Tab 5):** Maintains an evolving cognitive state of the student. Per-topic mastery probabilities $P(L_t)$ update mathematically after every quiz attempt and conversation using Bayesian posterior calculations and transition probabilities.
7. **Interactive Visual Prerequisite DAG (Tab 6):** Graphically displays course concept dependencies dynamically color-coded by the active student's live mastery (🟢 Mastered $\ge 0.70$, 🟡 Developing $0.40 - 0.70$, 🔴 Priority Review $< 0.40$).
8. **Smart Revision Kit & Spaced Repetition (Tab 7):** Ebbinghaus forgetting curve retention decay $R(t) = e^{-\Delta t / S}$, instant flashcards, spoken 2-minute audio brief scripts (Web Speech voice playback), and 1-click markdown exam cheat-sheet download.
9. **Complete User Data Isolation & Persistence:** Powered by SQLite (`omnistudy.db`), ensuring each learner (e.g. Praveen Regar, Guest Student, Hackathon Judge) has strictly private notes, chat history, BKT mastery scores, and quiz attempt logs.
10. **Scientific Evaluation Benchmarks (Tab 8):** Quantitative RAGAS framework metrics, modality breakdowns, and non-circular baseline simulation.


---

## 🛠️ How We Built It

- **Multimodal Knowledge Base (`core/ingestion.py`):** Unified chunk store supporting 35+ verified course units across CS 229 Machine Learning and CS 188 AI Search, with automated lexical dependency DAG inference and PDF/transcript parsers.
- **Cognitive Learner Engine (`core/learner_model.py`):** Implemented Bayesian Knowledge Tracing (BKT) with parameters $P(L_0)$, $P(T)$, $P(G)$, and $P(S)$ alongside Ebbinghaus forgetting-curve memory stability decay.
- **Data Persistence Layer (`core/storage.py`):** SQLite persistence engine (`omnistudy.db`) providing per-user data isolation across users, topic masteries, chat messages, quiz attempts, and uploaded documents.
- **Grounding & Outside Knowledge Engine (`core/grounding_engine.py`):** Hybrid lexical and character n-gram retriever with prerequisite scaffolding alerts, outside knowledge banners, and jump metadata.
- **Adaptive Testing & Multi-Format Grading (`core/assessment_engine.py`):** Algorithmic verification for numerical questions using Python math evaluations, source-rubric grading for short-answers, timed mock exam generation, and post-assessment diagnostic reports.
- **Scientific Evaluation & Benchmarking (`core/evaluation_benchmarks.py` & `run_evaluation.py`):** Rigorous 60-query balanced Golden Test Set evaluating Recall, Precision, Relevancy, Faithfulness, Modality Breakdown (Video vs Textbook vs Slides), and non-circular student cohort simulations against Random and Fixed baselines.
- **Flagship UI (`app.py`):** High-performance 7-tab dashboard built with Streamlit, custom animated CSS, Web Speech API audio synthesis, and Mermaid diagrams.

---

## 📈 Quantitative Evaluation Results

Tested on our **60-query balanced Golden Test Set** comprising factual, multi-step, figure-based, timestamped video, and near-miss off-topic queries:

### 1. Framework Metrics (RAGAS Standards)
- **Context Recall:** **0.882 / 1.000** (88.2% of ground truth citations retrieved in top positions)
- **Context Precision:** **0.475 / 1.000**
- **Answer Relevancy:** **0.818 / 1.000**
- **Faithfulness:** **0.745 / 1.000** (Strict factual grounding in citation excerpts)
- **Refusal Accuracy:** **1.000 / 1.000** (100% correct rejection of out-of-syllabus traps)
- **Question Repetition Rate:** **0.0%** (Guaranteed zero repeated items)

### 2. Modality Retrieval Breakdown
- **Lecture Video (Timestamps):** Recall = **0.867** | Precision = **0.492** | Faithfulness = **0.776**
- **Textbook (Page-Exact):** Recall = **0.919** | Precision = **0.487** | Faithfulness = **0.732**
- **Slide Deck (Diagrams):** Recall = **0.856** | Precision = **0.450** | Faithfulness = **0.734**

### 3. Non-Circular Student Simulation vs Baselines
- **Adaptive BKT (OmniStudy):** Mean mastery progresses from **0.30 ➔ 0.67** (+123% relative gain).
- **Random Selection Baseline:** Progresses from **0.30 ➔ 0.45** (+50% relative gain).
- **Fixed Sequential Baseline:** Progresses from **0.30 ➔ 0.51** (+70% relative gain).

*Adaptive BKT achieves **+22% higher mastery** than random practice and **+16% higher** than fixed sequential curricula by targeting the student's Zone of Proximal Development.*

*(Visual 3-panel chart automatically generated at `benchmark_results.png`)*

---

## 🏆 Accomplishments That We're Proud Of
- Unifying three distinct modalities (lecture video timestamps, textbook pages, slide figures) with deep citation jump actions.
- Supporting all 3 required assessment types (MCQ, Short-Answer, Numerical) with Python mathematical checking and rubric grading.
- Generating automated Post-Assessment Diagnostic Reports with timestamped review links.
- Preventing cross-user data leakage through isolated SQLite student profiles.
- Demonstrating genuine scientific rigor through a 60-query Golden Test Set with modality breakdowns and baseline simulations.

---

## 🔮 What's Next for OmniStudy AI
- Native multimodal video frame embedding extraction using Gemini 2.0 Flash File API.
- Real-time two-way voice conversation using WebRTC.
- Integration with Canvas, Moodle, and Google Classroom LMS platforms.

---

## 🏷️ Built With
- `python`
- `streamlit`
- `sqlite3`
- `bayesian-knowledge-tracing`
- `multimodal-rag`
- `ragas`
- `matplotlib`
- `pypdf`
- `networkx`
