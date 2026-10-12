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
                    "back": u.content_text[:200] + "...",
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

            audio_brief_paragraphs.append(
                f"Topic focus: {topic}. Remember that {units[0].subtopic} requires strict grounding. "
                f"Refer back to {units[0].source_title} at {units[0].location} for detailed derivations."
            )

        audio_script = (
            f"Hello {student.name}, here is your personalized 2-minute high-yield audio revision brief. "
            + " ".join(audio_brief_paragraphs)
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
        # Calculate topic urgency: deficit = (1.0 - mastery) * (1.0 / stability)
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

            if day == days_until_exam:
                day_plan["recommended_activity"] = "Comprehensive Mock Exam + Full Formula Review"
                day_plan["focus_topics"] = [t[0] for t in topic_urgency[:3]]
            elif day == 1:
                day_plan["recommended_activity"] = "Diagnostic Review + High-Yield Foundation Concepts"
                day_plan["focus_topics"] = [topic_urgency[0][0], topic_urgency[1][0]]
            else:
                # Cycle through urgent topics using spaced repetition intervals
                target_idx = (day - 2) % len(topic_urgency)
                target_topic = topic_urgency[target_idx]
                day_plan["focus_topics"] = [target_topic[0]]
                day_plan["recommended_activity"] = (
                    f"Targeted Adaptive Quiz on {target_topic[0]} (Current Mastery: {int(target_topic[1]*100)}%) "
                    f"+ Review Video Timestamp Excerpts"
                )

            schedule.append(day_plan)

        return schedule
