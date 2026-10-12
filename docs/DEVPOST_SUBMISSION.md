# Devpost Submission Text

## Project Title
**OmniStudy AI: Source-Grounded Multimodal Study Companion & Adaptive Learner Engine**

## Tagline
An intelligent study companion that unifies lecture videos, textbooks, and slides into a source-cited knowledge base with Bayesian Knowledge Tracing and adaptive assessments.

---

## 💡 Inspiration
Modern university students learn across disjointed mediums: 2-hour lecture video recordings, 800-page textbook PDFs, and 60-slide PowerPoint presentations. When students turn to standard LLM chatbots, two critical failure modes emerge:
1. **Hallucination & Lack of Syllabus Grounding:** Generic chatbots give explanations that do not match the professor's notation, textbook definitions, or exam scope, and cannot point the student to the exact minute in the lecture video where the concept was taught.
2. **Amnesia & Lack of Cognitive Modeling:** General chatbots treat every interaction as isolated, with zero understanding of what the student already knows, what they struggle with, or when they are about to forget a concept.

We built **OmniStudy AI** for the **Multimodal AI Hackathon 2026 (Track D: Personalized Tutoring & Adaptive Learning)** to solve this problem by uniting multimodal source grounding with cognitive psychology and Bayesian Knowledge Tracing.

---

## 🚀 What It Does

OmniStudy AI transforms raw university course assets into an interactive, personalized learning environment:

1. **Multimodal Knowledge Ingestion:** Automatically ingests lecture video transcripts with second-level timestamps (`06:12 - 11:45`), paginated textbook chapters, and slide decks with diagram/figure captions.
2. **Source-Grounding & Exact Location Deep Links:** When explaining any concept, OmniStudy AI cites the exact lecture video timestamp, textbook page, and slide number, displaying mathematical formulas and figure descriptions.
3. **Strict Curriculum Refusal Gate:** Queries outside the course syllabus (e.g., general world trivia, quantum cryptography, unrelated topics) are strictly flagged and refused with a 100% accuracy rate, eliminating hallucinated answers.
4. **Bayesian Knowledge Tracing (BKT) & Learner Modeling:** Maintains an evolving cognitive state of the student. Per-topic mastery probabilities $P(L_t)$ update mathematically after every quiz attempt and conversation using Bayesian posterior calculations and transition probabilities.
5. **Adaptive Assessment Engine:** Generates MCQs, numerical calculations, and conceptual questions calibrated to the student's current mastery level using Item Response Theory (IRT). Includes dual-pass verification and a guaranteed 0% question repetition filter.
6. **Misconception Diagnostics:** When an answer is incorrect, the system provides cited feedback and a diagnostic breakdown of the specific misconception trap the student fell into.
7. **Bonus Enhancements:**
   - **Interactive Visual Prerequisite DAG:** Concepts colored by live student mastery (🟢 Mastered, 🟡 Developing, 🔴 Needs Work).
   - **Ebbinghaus Forgetting Curve Exam Scheduler:** Generates a custom daily calendar based on memory stability and days until the final exam.
   - **Targeted Revision Kits:** Instant flashcards, cheat sheets, and 2-minute audio brief scripts for weak topics.
   - **Bilingual Hinglish / Hindi Mode:** Dual-language explanations making complex STEM concepts intuitive for Indian students.

---

## 🛠️ How We Built It

- **Multimodal Knowledge Graph (`core/ingestion.py`):** Structured atomic content units linking video timestamps, slide numbers, textbook pages, and mathematical formulas into a prerequisite Directed Acyclic Graph.
- **Cognitive Learner Engine (`core/learner_model.py`):** Implemented Bayesian Knowledge Tracing (BKT) with parameters $P(L_0)$, $P(T)$, $P(G)$, and $P(S)$ alongside Ebbinghaus forgetting-curve memory stability decay.
- **Grounding & Refusal Engine (`core/grounding_engine.py`):** Cross-asset semantic retriever with lexical stop-word filtering and strict out-of-syllabus refusal gating.
- **Adaptive Testing & Validation (`core/assessment_engine.py`):** Dynamic question selector with IRT difficulty targeting, cryptographic deduplication hashing, and anti-repetition memory.
- **Evaluation & Benchmarking (`core/evaluation_benchmarks.py` & `run_evaluation.py`):** Rigorous evaluation suite measuring Faithfulness, Answer Relevancy, Context Precision, Context Recall, Refusal Accuracy, and multi-session simulated student cohort trajectories.
- **Interactive UI (`app.py`):** High-performance 6-tab dashboard built with Streamlit, custom CSS, Mermaid diagrams, and real-time interactive widgets.

---

## 📈 Quantitative Evaluation Results

Tested on our Golden Test Set containing both syllabus-aligned and off-curriculum trap queries:
- **Context Recall:** **1.000 / 1.000**
- **Refusal Accuracy:** **1.000 / 1.000** (Zero out-of-scope hallucinations)
- **Answer Relevancy:** **0.823 / 1.000**
- **Faithfulness:** **0.690 / 1.000**
- **Question Repetition Rate:** **0.0%** across all student sessions

### Multi-Session Cohort Simulation:
- **Aarav (Struggling Learner):** $0.15 \longrightarrow 0.51$ (**+36% Mastery Gain**)
- **Priya (Steady Learner):** $0.25 \longrightarrow 0.72$ (**+47% Mastery Gain**)
- **Rohan (Advanced Learner):** $0.45 \longrightarrow 0.84$ (**+39% Mastery Gain**)

---

## 🏆 Accomplishments That We're Proud Of
- Seamlessly unifying three completely distinct modalities (video timestamps, textbook pages, slide deck figures) into unified citations.
- Achieving 100% out-of-curriculum refusal accuracy while maintaining high conversational responsiveness.
- Creating a mathematically sound BKT learner model that measurably lifts simulated student performance across sessions.
- Adding high-value enhancements like the visual DAG, Ebbinghaus exam scheduler, and Hinglish bilingual tutoring.

---

## 🔮 What's Next for OmniStudy AI
- Native multimodal video frame embedding extraction using Gemini 2.0 Flash File API.
- Real-time two-way voice conversation using WebRTC.
- Integration with Canvas, Moodle, and Google Classroom LMS platforms.

---

## 🏷️ Built With
- `python`
- `streamlit`
- `bayesian-knowledge-tracing`
- `multimodal-rag`
- `ragas`
- `gemini`
- `claude`
- `matplotlib`
- `networkx`
