# Demonstration Video Script (5-Minute Winning Pitch)

> **Track D: Personalized Tutoring & Adaptive Learning**  
> **Multimodal AI Hackathon 2026** (KamandPrompt, IIT Mandi, Augli.ai, PurpleRain Tech, LDV Labs)  
> **Target Video Duration:** 4 to 5 Minutes (YouTube Unlisted)

---

## 🎬 Video Production Overview

| Section | Timestamp | Visual on Screen | Speaker Script / Talking Points |
|---|---|---|---|
| **1. Hook & Problem** | 00:00 - 00:45 | Title Slide / Course Material Collage | The problem students face: fragmented materials, generic LLM hallucinations, and amnesic chatbots. |
| **2. Architecture & Ingestion** | 00:45 - 01:30 | Architecture Diagram & Ingestion Panel | How OmniStudy ingests video timestamps, textbook pages, and slide figures into a prerequisite DAG. |
| **3. Grounded Tutor & Refusal Gate** | 01:30 - 02:30 | Tab 1: Live Tutor Chat | Demo in-scope query with timestamp/page citations & diagram callouts; demo off-curriculum refusal. |
| **4. Cognitive Learner Model & Quizzes**| 02:30 - 03:30 | Tab 2 & Tab 3: Adaptive Testing & BKT | Live quiz attempt, BKT mathematical posterior update, misconception diagnosis, 0% repetition. |
| **5. Enhancements & Benchmarks** | 03:30 - 04:30 | Tab 4, 5, 6: DAG Graph, Schedule, RAGAS | Prerequisite DAG, Ebbinghaus exam scheduler, RAGAS benchmark metrics & student cohort gains. |
| **6. Conclusion & Vision** | 04:30 - 05:00 | Summary Slide & GitHub Link | Concluding thoughts, alignment with hackathon goals, and call to action. |

---

## 🎙️ Minute-by-Minute Speaker Script

### [00:00 - 00:45] 1. The Core Problem
**[Visual: Title Slide with OmniStudy AI Logo, then show tabs open with YouTube lecture video, a 500-page textbook PDF, and a slide deck.]**

> *"Hello judges! Welcome to our demonstration of **OmniStudy AI**, built for Track D of the Multimodal AI Hackathon 2026.*  
> *Every university student faces a major hurdle: course knowledge is scattered across hour-long lecture videos, dense textbook chapters, and slide decks. When students turn to standard chatbots like ChatGPT or Claude, they face two critical issues:*  
> *First: hallucinations. Standard LLMs give answers disconnected from the professor's curriculum and can never cite the exact timestamp in the lecture video.*  
> *Second: cognitive blindness. They have zero memory of what a student knows, what they struggle with, or when they are about to forget.*  
> *OmniStudy AI solves this by combining multimodal source grounding with Bayesian Knowledge Tracing."*

---

### [00:45 - 01:30] 2. System Architecture & Multimodal Ingestion
**[Visual: Show System Architecture Flowchart, then switch to the Streamlit app sidebar.]**

> *"Let's look at the architecture. OmniStudy ingests three distinct modalities without manual preprocessing:*  
> *1. Lecture videos with second-level timestamps.*  
> *2. Textbooks with page-level extractions.*  
> *3. Slide decks with embedded diagram captions.*  
> *As you can see in the sidebar, we have pre-indexed a rigorous university course: CS 229 Machine Learning. Users can also dynamically ingest custom lecture notes or transcripts with one click.*  
> *Every atomic unit is tagged with its origin, topic, and prerequisite dependencies, building a unified knowledge graph."*

---

### [01:30 - 02:30] 3. Source-Grounded Tutor & Strict Refusal Gate
**[Visual: Navigate to Tab 1: 'Source-Grounded Tutor'. Click on the prompt 'Why 1/2 factor in MSE?'.]**

> *"Now let's test our Grounded AI Tutor. I'll ask: 'Why is the 1/2 factor included in the linear regression cost function?'*  
> *Notice what happens: OmniStudy doesn't just give an answer. It provides the exact mathematical formulation, quotes Lecture 1 at timestamp 06:12 to 11:45, cross-references Slide 8, and highlights Figure 1: the scatter plot residuals.*  
> *Now, watch our strict Refusal Gate in action. If a student asks an off-curriculum question—like 'How does quantum key distribution protocol BB84 work?'—OmniStudy does NOT hallucinate. It immediately flags the query as out-of-curriculum and refuses to generate ungrounded claims, ensuring 100% syllabus integrity."*

---

### [02:30 - 03:30] 4. Bayesian Knowledge Tracing & Adaptive Assessment
**[Visual: Navigate to Tab 2: 'Adaptive Assessments'. Select 'Supervised Learning' and click 'Generate Adaptive Quiz'.]**

> *"Next, let's explore our Cognitive Learner Model and Adaptive Assessment Engine.*  
> *We have implemented Bayesian Knowledge Tracing—or BKT—combined with Item Response Theory. The system models four key probabilities for each concept: prior knowledge P(L0), learning transition P(T), guess rate P(G), and slip rate P(S).*  
> *Notice the question generated: it is dynamically calibrated to our active student, Priya's, current mastery level.*  
> *Let's select an answer and submit. Immediately, we get:*  
> *1. Exact cited feedback with source location.*  
> *2. A misconception diagnostic explaining why alternative options are traps.*  
> *3. A live Bayesian update: Priya's mastery in Supervised Learning just increased from 0.40 to 0.58!*  
> *And our cryptographic deduplication filter guarantees a 0% question repetition rate across all assessment sessions."*

---

### [03:30 - 04:30] 5. Visual Prerequisite DAG, Revision Kits & RAGAS Benchmarks
**[Visual: Click through Tab 4: Prerequisite DAG, Tab 5: Revision Kit, and Tab 6: Benchmarks.]**

> *"We also built all optional enhancements requested in Track D:*  
> *In Tab 4, you can see our interactive visual Prerequisite DAG. Notice the node coloring: concepts turn green when mastered (P(L) >= 0.75), yellow when developing, and red when urgent remediation is required.*  
> *In Tab 5, OmniStudy auto-generates personalized Flashcards for weak topics, a 2-minute audio revision brief script, and an exam study calendar calculated using the Ebbinghaus Forgetting Curve.*  
> *Finally, in Tab 6, we report our rigorous system benchmarks:*  
> *Using RAGAS evaluation criteria on our golden test set:*  
> *Context Recall: 1.000.*  
> *Refusal Accuracy: 1.000.*  
> *Answer Relevancy: 0.823.*  
> *And in our multi-session simulation across struggling, steady, and advanced student profiles, all students achieved measurable mastery gains between 36% and 47%.*"

---

### [04:30 - 05:00] 6. Conclusion & GitHub Code
**[Visual: Show GitHub README and final summary screen.]**

> *"OmniStudy AI demonstrates that by combining multimodal source grounding with cognitive learner modeling, we can make AI tutoring trustworthy, syllabus-grounded, and genuinely adaptive.*  
> *The entire codebase is open-source, fully documented, and ready to run with a single command.*  
> *Thank you to KamandPrompt at IIT Mandi, Augli.ai, PurpleRain Tech, and LDV Labs for this incredible hackathon challenge!"*

---

## 💡 Tips for Recording the YouTube Video:
1. **Screen Resolution:** Record at 1080p (1920x1080) at 60fps using OBS Studio or Windows Game Bar (`Win + G`).
2. **Audio Quality:** Use a decent USB microphone or headset in a quiet room.
3. **Pacing:** Follow the script cues closely to keep the video between 4:00 and 4:45.
4. **YouTube Settings:** Upload as **Unlisted** (as specified in guidelines: *"Unlisted is fine. Private is not"*). Ensure title matches Devpost project title.
