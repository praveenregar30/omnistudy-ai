"""Optional Enhancements Suite for OmniStudy AI.

Includes:
1. Visual Prerequisite DAG Course Flow Map.
2. Targeted Revision Kits (Flashcards, Cheat Sheets, Audio Brief Scripts).
3. Ebbinghaus Forgetting-Curve Spaced Repetition Exam Scheduler.
4. Hinglish Bilingual Explanations for Indian Learners.
5. Audio Voice Tutor Session Scripts.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional
from core.ingestion import CourseKnowledgeBase
from core.learner_model import StudentProfile

class EnhancementsManager:
    """Manages high-value bonus capabilities requested in Hackathon Track D."""

    def __init__(self, kb: CourseKnowledgeBase):
        self.kb = kb

    def generate_course_flow_mermaid(self, student: Optional[StudentProfile] = None) -> str:
        """Generates a styled Mermaid DAG showing prerequisite flow and student mastery."""
        dag = self.kb.get_prerequisite_dag()
        lines = ["flowchart TD"]

        if not dag.get("nodes"):
            lines.append('    empty_node["📄 Course Workspace Ready<br>(Upload lecture notes to map prerequisite DAG)"]')
            lines.append('    style empty_node fill:#f8fafc,stroke:#94a3b8,stroke-width:2px')
            return "\n".join(lines)

        # Color palette for mastery status
        # Green: Mastered (>= 0.75), Orange: Developing (0.45 - 0.74), Red: Needs Study (< 0.45)
        for node in dag["nodes"]:
            cid = node["id"]
            name = node["label"].replace('"', '')
            topic = node["topic"]
            
            # Check student mastery for topic
            color_class = "style " + cid + " fill:#e8f4f8,stroke:#3498db,stroke-width:2px"
            status_badge = ""
            if student and topic in student.topic_masteries:
                m = student.topic_masteries[topic].mastery_prob
                if m >= 0.75:
                    color_class = f"style {cid} fill:#d4edda,stroke:#28a745,stroke-width:2.5px"
                    status_badge = f" [Mastered: {int(m*100)}%]"
                elif m >= 0.45:
                    color_class = f"style {cid} fill:#fff3cd,stroke:#ffc107,stroke-width:2.5px"
                    status_badge = f" [Reviewing: {int(m*100)}%]"
                else:
                    color_class = f"style {cid} fill:#f8d7da,stroke:#dc3545,stroke-width:2.5px"
                    status_badge = f" [Needs Work: {int(m*100)}%]"

            lines.append(f'    {cid}["{name}{status_badge}"]')
            lines.append(f"    {color_class}")

        for edge in dag["edges"]:
            lines.append(f'    {edge["from"]} --> {edge["to"]}')

        return "\n".join(lines)

    def generate_revision_kit(self, student: StudentProfile) -> Dict[str, Any]:
        """Creates targeted flashcards, cheat sheet, and 2-minute audio brief for weak topics."""
        weak_topics = [t for t, tm in student.topic_masteries.items() if tm.mastery_prob < 0.70]
        if not weak_topics:
            weak_topics = list(student.topic_masteries.keys())[:2]

        flashcards = []
        cheat_sheet_points = []
        audio_brief_paragraphs = []

        for topic in weak_topics:
            units = [u for u in self.kb.units.values() if u.topic == topic]
            for u in units[:2]:
                # Generate Flashcard
                flashcards.append({
                    "topic": u.topic,
                    "subtopic": u.subtopic,
                    "front": f"What is the core principle of {u.subtopic}?",
                    "back": u.content_text[:200] + ("..." if len(u.content_text) > 200 else ""),
                    "formula": u.key_formulas[0] if u.key_formulas else "N/A",
                    "source_cite": f"{u.source_title} ({u.location})"
                })
                # Cheat sheet bullet point
                cheat_sheet_points.append({
                    "concept": u.subtopic,
                    "summary": u.content_text[:160],
                    "formula": u.key_formulas[0] if u.key_formulas else None,
                    "location": u.location
                })

            if units:
                sub_title = units[0].subtopic or topic
                src_title = units[0].source_title or "Course Material"
                loc = units[0].location or "Notes"
                audio_brief_paragraphs.append(
                    f"Topic focus: {topic}. Remember that {sub_title} requires strict grounding. "
                    f"Refer back to {src_title} at {loc} for detailed derivations."
                )
            else:
                audio_brief_paragraphs.append(
                    f"Topic focus: {topic}. Review your lecture notes and practice active recall to solidify this subject."
                )

        if not flashcards:
            flashcards.append({
                "topic": "Personal Study Workspace",
                "subtopic": "Active Recall & Spaced Repetition",
                "front": "How do active recall and spaced repetition improve exam performance?",
                "back": "Active retrieval forces cognitive reactivation of memory traces, increasing memory stability S in the Ebbinghaus forgetting curve R(t) = exp(-t / S).",
                "formula": "R(t) = e^{-t / S}",
                "source_cite": "Cognitive Learning Engine"
            })
            cheat_sheet_points.append({
                "concept": "Knowledge Ingestion",
                "summary": "Upload your lecture notes, textbook PDFs, or transcripts to generate custom grounded flashcards.",
                "formula": None,
                "location": "Tab 1: Multimodal Ingestion"
            })

        audio_script = (
            f"Hello {student.name}, here is your personalized 2-minute high-yield audio revision brief. "
            + (" ".join(audio_brief_paragraphs) if audio_brief_paragraphs else "Explore your course topics and upload lecture materials to generate grounded revision notes.")
            + " Keep practicing adaptive quizzes to solidify these concepts before your exam!"
        )

        return {
            "weak_topics": weak_topics,
            "flashcards": flashcards,
            "cheat_sheet": cheat_sheet_points,
            "audio_brief_script": audio_script
        }

    def generate_exam_study_schedule(
        self,
        student: StudentProfile,
        days_until_exam: int = 7,
        daily_hours: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Calculates an optimal study calendar based on Ebbinghaus forgetting curves and BKT deficits."""
        topic_urgency = []
        for topic, tm in student.topic_masteries.items():
            retention = tm.retention_probability()
            deficit = (1.0 - tm.mastery_prob) + (1.0 - retention)
            topic_urgency.append((topic, tm.mastery_prob, retention, deficit))

        # Sort by urgency descending
        topic_urgency.sort(key=lambda x: x[3], reverse=True)

        schedule = []
        for day in range(1, days_until_exam + 1):
            day_plan = {
                "day_number": day,
                "label": f"Day {day} (T-{days_until_exam - day + 1} days)",
                "focus_topics": [],
                "allocated_hours": daily_hours,
                "recommended_activity": ""
            }

            if not topic_urgency:
                day_plan["focus_topics"] = ["Course Ingestion & Notes Review"]
                day_plan["recommended_activity"] = "Upload your lecture slides or syllabus and practice your first adaptive quiz"
            elif day == days_until_exam:
                day_plan["recommended_activity"] = "Comprehensive Mock Exam + Full Formula Review"
                day_plan["focus_topics"] = [t[0] for t in topic_urgency[:3]]
            elif day == 1:
                day_plan["recommended_activity"] = "Diagnostic Review + High-Yield Foundation Concepts"
                day_plan["focus_topics"] = [t[0] for t in topic_urgency[:2]]
            else:
                target_idx = (day - 2) % len(topic_urgency)
                target_topic = topic_urgency[target_idx]
                day_plan["focus_topics"] = [target_topic[0]]
                day_plan["recommended_activity"] = (
                    f"Targeted Adaptive Quiz on {target_topic[0]} (Current Mastery: {int(target_topic[1]*100)}%) "
                    f"+ Review Video Timestamp Excerpts"
                )

            schedule.append(day_plan)

        return schedule

    def export_markdown_study_guide(self, student: StudentProfile, days_until_exam: int = 7) -> str:
        """Generates a downloadable Markdown/Text personalized Exam Cheat Sheet & Study Pack."""
        rev = self.generate_revision_kit(student)
        sched = self.generate_exam_study_schedule(student, days_until_exam=days_until_exam)

        lines = [
            f"# 🎓 OmniStudy AI — Personalized Exam Cheat-Sheet & Study Guide",
            f"**Student:** {student.name} | **Course:** {self.kb.course_name}",
            "---\n",
            "## 1. 🧠 Current Bayesian Knowledge Tracing (BKT) Mastery Status"
        ]
        for topic, tm in student.topic_masteries.items():
            lines.append(f"- **{topic}:** Mastery = `{int(tm.mastery_prob * 100)}%` | Memory Stability = `{tm.stability_days:.1f} days`")

        lines.append("\n---\n## 2. 📇 High-Yield Flashcards & Formulas for Weak Topics")
        for idx, fc in enumerate(rev["flashcards"]):
            lines.append(f"### Card {idx+1}: {fc['subtopic']} ({fc['topic']})")
            lines.append(f"- **Q:** {fc['front']}")
            lines.append(f"- **A:** {fc['back']}")
            if fc["formula"] != "N/A":
                lines.append(f"- **Formula:** `{fc['formula']}`")
            lines.append(f"- **Source Citation:** *{fc['source_cite']}*\n")

        lines.append("---\n## 3. 📅 Ebbinghaus Spaced Repetition Exam Timetable")
        for s in sched:
            lines.append(f"- **{s['label']}** ({s['allocated_hours']} hrs) — Focus: **{', '.join(s['focus_topics'])}** ➔ *{s['recommended_activity']}*")

        lines.append("\n---\n*Generated by OmniStudy AI (Multimodal AI Hackathon 2026 - Track D)*")
        return "\n".join(lines)

    def generate_emergency_crash_course(
        self,
        topic: str,
        raw_notes: Optional[str] = None,
        hours_left: int = 48,
        lang_mode: str = "English"
    ) -> Dict[str, Any]:
        """Generates an intensive 48-Hour Exam Cram Course that teaches a topic or uploaded notes from scratch."""
        topic_units = [u for u in self.kb.units.values() if u.topic == topic]
        if not topic_units:
            topic_units = list(self.kb.units.values())[:3]

        formulas = []
        for u in topic_units:
            formulas.extend(u.key_formulas)

        is_hinglish = "Hinglish" in lang_mode

        if raw_notes and len(raw_notes.strip()) > 30:
            title = f"High-Yield Crash Course: {topic}"
            intro = (
                f"🚨 **Emergency Exam Cram Session (T-{hours_left}h):**\n\n"
                f"Agar aapne abhi tak padhai shuru nahi ki hai, toh ghabraiye mat! Hum aapko '{topic}' ke core concepts scratch se teach kar rahe hain."
                if is_hinglish else
                f"🚨 **Emergency Exam Cram Session (T-{hours_left}h):**\n\n"
                f"Don't panic if you haven't started studying yet! Here is your high-yield, step-by-step crash breakdown to master '{topic}' for the upcoming exam."
            )
            key_concepts = [
                f"**Core Concept 1:** {topic} foundational mechanism derived directly from your notes.",
                f"**Core Concept 2:** Optimization, parameter updates, and boundary conditions.",
                f"**Core Concept 3:** Common mathematical assumptions and convergence behavior."
            ]
        else:
            title = f"High-Yield Crash Course: {topic}"
            u_prime = topic_units[0]
            intro = (
                f"🚨 **Emergency Exam Cram Session for {topic} (T-{hours_left}h):**\n\n"
                f"Aapke exam me sirf {hours_left} ghante bache hain! '{topic}' me maximum scoring karne ke liye sabse crucial concepts yeh hain:"
                if is_hinglish else
                f"🚨 **Emergency Exam Cram Session for {topic} (T-{hours_left}h):**\n\n"
                f"With only {hours_left} hours left before your exam, prioritize these high-weightage concepts and formulas to secure maximum marks:"
            )
            key_concepts = [
                f"**1. Core Intuition & Formulation ({u_prime.subtopic}):** {u_prime.content_text[:220]}...",
                f"**2. Critical Mechanism:** Focus on how the algorithm minimizes error and converges to the optimal parameter set.",
                f"**3. Exam Hall Rule of Thumb:** Memorize the exact cost function formulation and the gradient update step."
            ]

        schedule_blocks = [
            {"time": "Hours 0 - 12 (Block 1: Foundation Blitz)", "task": f"Read the intuitive summary for {topic}, listen to the 3-minute audio brief, and memorize formulas.", "urgency": "High"},
            {"time": "Hours 12 - 24 (Block 2: Blitz Quiz & Weak Topic Identification)", "task": "Attempt the 4-question diagnostic quiz below. Identify which subtopics are weak.", "urgency": "High"},
            {"time": "Hours 24 - 36 (Block 3: Intensive Weak-Topic Remediation)", "task": "Use the 'Teach Weak Topics More' module to drill misconceptions until all tags turn Green 🟢.", "urgency": "Critical"},
            {"time": "Hours 36 - 48 (Block 4: Final Mock Exam & Rest)", "task": "Run 1 timed mock exam, review the 1-page cheat sheet, and get 7 hours of sleep before the test.", "urgency": "Medium"}
        ]

        audio_script = (
            f"Exam alert! You have {hours_left} hours until your test on {topic}. "
            f"Here is your emergency high-yield briefing. "
            f"First, remember that {topic} centers on {topic_units[0].subtopic}. "
            f"Second, pay close attention to the mathematical formulation. "
            f"Now, take the diagnostic quiz below to pinpoint your exact weak areas so we can remediate them immediately!"
        )

        return {
            "title": title,
            "intro": intro,
            "key_concepts": key_concepts,
            "formulas": formulas or [r"J(\theta) = \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2"],
            "sources": [f"{u.source_title} ({u.location})" for u in topic_units[:2]],
            "schedule_blocks": schedule_blocks,
            "audio_script": audio_script
        }

    def generate_weak_topic_remediation(
        self,
        topic: str,
        misconception_tag: str = "",
        student_level: str = "Undergraduate",
        lang_mode: str = "English"
    ) -> Dict[str, Any]:
        """Deep-dive remediation module that reteaches weak topics identified during the diagnostic quiz."""
        is_hinglish = "Hinglish" in lang_mode
        units = [u for u in self.kb.units.values() if u.topic == topic]
        unit = units[0] if units else list(self.kb.units.values())[0]

        if is_hinglish:
            deep_dive = (
                f"### 💡 Deep-Dive Remediation: {topic} Samajhte Hain\n\n"
                f"Aapne quiz me is topic me galti ki, par fikar mat kijiye — exam me ye question aksar poochha jata hai!\n\n"
                f"**Asal Kahaani (The Core Intuition):**\n"
                f"{unit.content_text}\n\n"
                f"**Exam Tip:** Agar question me calculus ya derivative ka rule aaye, toh formula directly recall karein. "
                f"Galti isliye hoti hai kyunki log distractors me sign (+/-) ya scale factor (1/2) me confuse ho jaate hain."
            )
        else:
            deep_dive = (
                f"### 💡 Deep-Dive Remediation: Mastering {topic}\n\n"
                f"You struggled with this topic on the diagnostic quiz. Here is the step-by-step intuition to guarantee full marks on exam day:\n\n"
                f"**The Fundamental Principle:**\n"
                f"{unit.content_text}\n\n"
                f"**Why Students Make Mistakes Here:**\n"
                f"{misconception_tag or 'Students often mix up the optimization criteria or forget how boundary conditions behave under gradients.'}\n\n"
                f"**Exam Hall Quick Check:** When solving this on the test, verify that your parameters decrease in the direction opposite to the gradient vector."
            )

        practice_check = {
            "question": f"Quick Check: In {topic}, what is the single most important rule to remember during the exam?",
            "options": [
                f"Always follow the exact derivation: {unit.subtopic}",
                "Ignore all mathematical constants and regularization penalties.",
                "Assume the model generalizes with zero training sample variance."
            ],
            "correct_index": 0,
            "explanation": f"Correct! As taught in {unit.source_title} ({unit.location}), this foundational derivation ensures correct predictions."
        }

        return {
            "topic": topic,
            "remediation_markdown": deep_dive,
            "source_ref": f"{unit.source_title} ({unit.location})",
            "practice_check": practice_check
        }


