"""Source Grounding, Adaptive Scaffolding, & Outside-Knowledge Flagging Engine.

Implements:
1. Dual-mode handling: in-syllabus source-cited explanations vs. clearly segregated
   out-of-syllabus knowledge with disclaimers.
2. Cognitive scaffolding adapting explanation depth to student BKT mastery.
3. Interactive citation jump metadata (video seek seconds, PDF page, slide numbers).
4. Proactive prerequisite suggestions when foundational concepts are weak.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple, Optional
from core.ingestion import CourseKnowledgeBase, ContentUnit
from core.learner_model import StudentProfile

class GroundingEngine:
    """Manages grounded response synthesis, curriculum compliance, and adaptive scaffolding."""

    def __init__(self, knowledge_base: CourseKnowledgeBase):
        self.kb = knowledge_base

    def verify_query_scope(self, query: str) -> Tuple[bool, float, List[ContentUnit]]:
        """Determines if query is covered by course material using hybrid lexical + semantic overlap."""
        results = self.kb.search(query, top_k=4)
        if not results:
            return False, 0.05, []

        STOP_WORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "how", "what", "why", "when", "where", "who", "which", "does", "did",
            "do", "in", "of", "and", "to", "for", "with", "on", "at", "by", "from",
            "it", "this", "that", "these", "those", "or", "as", "if", "then", "so"
        }
        query_terms = set(re.findall(r"\w+", query.lower())) - STOP_WORDS
        top_unit = results[0]
        combined = f"{top_unit.topic} {top_unit.subtopic} {top_unit.content_text}".lower()
        unit_terms = set(re.findall(r"\w+", combined)) - STOP_WORDS
        
        overlap = len(query_terms.intersection(unit_terms))
        curriculum_keywords = {
            "regression", "classification", "gradient", "descent", "loss", "cost",
            "mse", "sigmoid", "logistic", "precision", "recall", "f1", "confusion",
            "bias", "variance", "regularization", "lasso", "ridge", "l1", "l2",
            "neural", "network", "backpropagation", "perceptron", "activation",
            "relu", "kmeans", "clustering", "centroid", "unsupervised", "supervised",
            "overfitting", "underfitting", "hyperplane", "convex", "learning", "rate",
            "search", "heuristics", "astar", "minimax", "alpha", "beta", "mdp", "qlearning",
            "markov", "reinforcement", "bellman", "frontier", "admissible", "consistent"
        }
        has_curriculum_kw = any(kw in query.lower() for kw in curriculum_keywords)
        is_custom_match = (top_unit.unit_id.startswith("custom_") and overlap >= 1)

        if overlap >= 2 or has_curriculum_kw or is_custom_match:
            confidence = min(0.96, 0.45 + (0.12 * min(4, max(1, overlap))))
            return True, confidence, results

        return False, 0.08, []

    def answer_query(
        self,
        query: str,
        student: Optional[StudentProfile] = None,
        student_level: str = "Undergraduate",
        language_mode: str = "English",
        allow_outside_knowledge: bool = True,
        gemini_api_key: str = "",
        **kwargs: Any
    ) -> Dict[str, Any]:

        """Synthesizes a source-cited explanation or cleanly segregates outside knowledge."""
        is_in_scope, confidence, retrieved_units = self.verify_query_scope(query)

        # 1. OUT-OF-SCOPE HANDLING (Requirement 2b & 6)
        if not is_in_scope:
            if not allow_outside_knowledge:
                response_text = (
                    "🛡️ **[Curriculum Guard: Out-of-Syllabus Query Declined]**\n\n"
                    "This topic is outside the official scope of your course materials "
                    f"(*{self.kb.course_name}*).\n\n"
                    "**Strict Grounding Policy:** As your grounded study companion, outside claims are blocked "
                    "to maintain 100% exam curriculum integrity.\n\n"
                    "*Toggle 'Allow Outside Knowledge Mode' if you want a general AI explanation clearly labeled as unverified.*"
                )
                return {
                    "in_scope": False,
                    "confidence": confidence,
                    "explanation": response_text,
                    "citations": [],
                    "primary_topic": None,
                    "is_outside_knowledge": True
                }
            else:
                # Provide outside knowledge with explicit visual segregation banner
                outside_explanation = (
                    "⚠️ **[NOT FROM YOUR COURSE MATERIAL — GENERAL AI KNOWLEDGE]**\n\n"
                    "> **Disclaimer:** The following answer is provided from general external knowledge. "
                    "**It is NOT present in your course slides, textbooks, or lecture videos**, and is "
                    "not guaranteed to match your instructor's exam syllabus.\n\n"
                    f"**General Answer to \"{query}\":**\n\n"
                    f"In broader science/technology, this topic involves general computational principles. "
                    f"Unlike topics in {self.kb.course_name}, this concept does not have dedicated course "
                    f"slides or video recordings in your currently indexed repository.\n\n"
                    "💡 *To return to verified curriculum content, ask questions about Linear Models, "
                    "Optimization, Classification, Neural Networks, Clustering, or AI Search.*"
                )
                return {
                    "in_scope": False,
                    "confidence": confidence,
                    "explanation": outside_explanation,
                    "citations": [],
                    "primary_topic": None,
                    "is_outside_knowledge": True
                }

        # 2. IN-SCOPE GROUNDED EXPLANATION
        primary_unit = retrieved_units[0]
        supporting_units = retrieved_units[1:]

        citations = []
        for u in retrieved_units:
            citations.append({
                "unit_id": u.unit_id,
                "source_type": u.source_type,
                "source_title": u.source_title,
                "location": u.location,
                "page": u.page_number,
                "slide": u.slide_number,
                "timestamp_sec": u.timestamp_seconds,
                "topic": u.topic,
                "subtopic": u.subtopic,
                "excerpt": u.content_text,
                "diagram": u.diagram.caption if u.diagram else None,
                # Jump viewer metadata for interactive opening
                "jump_type": "video_seek" if u.source_type == "lecture_video" else ("pdf_page" if u.source_type == "textbook" else "slide_view"),
                "jump_value": u.timestamp_seconds if u.source_type == "lecture_video" else (u.page_number or u.slide_number or 1)
            })

        # Adaptive Scaffolding Check based on Student's Mastery
        scaffolding_prefix = ""
        prereq_alert = ""
        if student and primary_unit.topic in student.topic_masteries:
            mastery = student.topic_masteries[primary_unit.topic].mastery_prob
            if mastery < 0.45:
                scaffolding_prefix = (
                    f"💡 *[Personalized Scaffolding for {student.name.split()[0]} (Mastery: {int(mastery*100)}%)]*\n"
                    f"Because your mastery in **{primary_unit.topic}** is currently developing, "
                    f"here is an intuitive, step-by-step conceptual walkthrough:\n\n"
                )
                # Check for weak prerequisites in DAG
                for cid, cnode in self.kb.concepts.items():
                    if primary_unit.subtopic.lower() in cnode.name.lower():
                        for pid in cnode.prerequisites:
                            p_concept = self.kb.concepts.get(pid)
                            if p_concept and p_concept.topic in student.topic_masteries:
                                p_m = student.topic_masteries[p_concept.topic].mastery_prob
                                if p_m < 0.50:
                                    prereq_alert = (
                                        f"> 📌 **Prerequisite Recommendation:** You may find this concept easier after "
                                        f"reviewing **{p_concept.name}** (Current Mastery: {int(p_m*100)}%).\n\n"
                                    )

        # Build Explanation
        explanation_blocks = []
        if prereq_alert:
            explanation_blocks.append(prereq_alert)
        if scaffolding_prefix:
            explanation_blocks.append(scaffolding_prefix)

        # Live Gemini LLM Synthesis option
        gemini_text = ""
        if gemini_api_key.strip():
            try:
                import requests
                context_blob = "\n".join([f"[{u.source_title} | {u.location}]: {u.content_text}" for u in retrieved_units])
                m_prob = student.topic_masteries[primary_unit.topic].mastery_prob if student and primary_unit.topic in student.topic_masteries else 0.5
                prompt = (
                    f"You are OmniStudy AI, an adaptive source-grounded university tutor. Answer the student's question "
                    f"strictly using the following course material context. Student mastery: {m_prob:.2f}. "
                    f"Language: {language_mode}. Cite exact locations in brackets.\n\n"
                    f"Course Context:\n{context_blob}\n\nQuestion: {query}"
                )
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key.strip()}"
                resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    gemini_text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                gemini_text = ""

        if gemini_text:
            explanation_blocks.append(f"### 🤖 Grounded Explanation: {primary_unit.subtopic}\n\n{gemini_text}\n")
        else:
            if language_mode == "Hinglish":
                explanation_blocks.append(
                    f"### 📘 {primary_unit.subtopic} (Grounded Breakdown)\n\n"
                    f"Course material ke mutabiq, **{primary_unit.subtopic}** ek core concept hai in **{primary_unit.topic}**."
                )
            else:
                explanation_blocks.append(
                    f"### 📘 {primary_unit.subtopic}\n\n"
                    f"According to the verified course material for **{primary_unit.topic}**:"
                )

            explanation_blocks.append(f"\n{primary_unit.content_text}\n")

        # Include Mathematical Formulation if present
        if primary_unit.key_formulas:
            formula_md = "\n**Key Mathematical Formulation:**\n"
            for f in primary_unit.key_formulas:
                formula_md += f"- $${f}$$\n"
            explanation_blocks.append(formula_md)

        # Include Visual Diagram
        if primary_unit.diagram:
            fig = primary_unit.diagram
            explanation_blocks.append(
                f"\n> 🖼️ **Associated Visual Material [{primary_unit.location}]:**\n"
                f"> **{fig.figure_id}** — *{fig.caption}*\n"
                f"> Context: {fig.context_text}\n"
            )

        # Cross-Modal Perspectives
        if supporting_units:
            cross_modal_text = "\n### 🔗 Cross-Modal Connections Across Course Assets:\n"
            for supp in supporting_units[:2]:
                icon = "🎥" if supp.source_type == "lecture_video" else ("📖" if supp.source_type == "textbook" else "📊")
                cross_modal_text += (
                    f"- {icon} **{supp.source_title}** (`{supp.location}`):\n"
                    f"  *{supp.content_text[:160]}...*\n"
                )
            explanation_blocks.append(cross_modal_text)

        if language_mode == "Hinglish" and not gemini_text:
            explanation_blocks.append(
                "\n💡 **Intuitive Summary (Hinglish):**\n"
                "Asaan lafzon mein: Yeh concept ensure karta hai ki model training data se seekh kar unseen "
                "data par accurate predictions de. Citations click karke aap exact video timestamp ya book page open kar sakte hain."
            )

        full_explanation = "\n".join(explanation_blocks)

        return {
            "in_scope": True,
            "confidence": confidence,
            "explanation": full_explanation,
            "plain_speech_text": primary_unit.content_text,
            "citations": citations,
            "primary_topic": primary_unit.topic,
            "primary_subtopic": primary_unit.subtopic,
            "is_outside_knowledge": False
        }
