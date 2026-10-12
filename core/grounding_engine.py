"""Strict Source Grounding & Refusal Gate Engine.

Implements source-cited concept explanation, exact location referencing
(page, slide number, video timestamp), and strict out-of-syllabus refusal/flagging.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
from core.ingestion import CourseKnowledgeBase, ContentUnit

class GroundingEngine:
    """Manages grounded response synthesis and curriculum compliance checking."""

    def __init__(self, knowledge_base: CourseKnowledgeBase):
        self.kb = knowledge_base

    def verify_query_scope(self, query: str) -> Tuple[bool, float, List[ContentUnit]]:
        """Determines if query is covered by course material.
        
        Returns (is_in_scope, confidence_score, matching_units).
        """
        results = self.kb.search(query, top_k=4)
        if not results:
            return False, 0.0, []

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
        # Known course keywords
        curriculum_keywords = {
            "regression", "classification", "gradient", "descent", "loss", "cost",
            "mse", "sigmoid", "logistic", "precision", "recall", "f1", "confusion",
            "bias", "variance", "regularization", "lasso", "ridge", "l1", "l2",
            "neural", "network", "backpropagation", "perceptron", "activation",
            "relu", "kmeans", "clustering", "centroid", "unsupervised", "supervised",
            "overfitting", "underfitting", "hyperplane", "convex", "learning", "rate",
            "clustering", "elbow", "matrix", "error", "bishop", "hastie", "murphy"
        }
        has_curriculum_kw = any(kw in query.lower() for kw in curriculum_keywords)

        if overlap >= 2 or has_curriculum_kw:
            confidence = min(0.98, 0.40 + (0.15 * min(4, overlap)))
            return True, confidence, results

        return False, 0.05, []

    def answer_query(
        self,
        query: str,
        student_level: str = "Undergraduate",
        language_mode: str = "English"  # 'English' or 'Hinglish'
    ) -> Dict[str, Any]:
        """Synthesizes a source-cited explanation or flags out-of-syllabus queries."""
        is_in_scope, confidence, retrieved_units = self.verify_query_scope(query)

        # 1. OUT-OF-SCOPE REFUSAL / SEGREGATION
        if not is_in_scope:
            response_text = (
                "⚠️ **[Off-Curriculum Alert: Query Not Covered in Course Materials]**\n\n"
                "This topic is outside the official scope of the ingested course materials "
                f"(*{self.kb.course_name}*).\n\n"
                "**Strict Grounding Policy:** As your grounded study companion, I cannot generate "
                "ungrounded claims that are not backed by your textbooks, slides, or lecture videos.\n\n"
                "*Tip: Try asking about Linear Regression, Gradient Descent, Logistic Regression, "
                "Confusion Matrix & Metrics, Bias-Variance Tradeoff, L1/L2 Regularization, "
                "Neural Networks/Backpropagation, or K-Means Clustering.*"
            )
            return {
                "in_scope": False,
                "confidence": confidence,
                "explanation": response_text,
                "citations": [],
                "primary_topic": None,
                "refusal_reason": "Query does not match any lecture video, textbook chapter, or slide deck."
            }

        # 2. IN-SCOPE SOURCE-CITED EXPLANATION
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
                "excerpt": u.content_text[:220] + "..." if len(u.content_text) > 220 else u.content_text,
                "diagram": u.diagram.caption if u.diagram else None
            })

        # Synthesize source-grounded response
        explanation_blocks = []
        
        # Primary Definition & Mathematical Formulation
        if language_mode == "Hinglish":
            explanation_blocks.append(
                f"### 📘 {primary_unit.subtopic} (Concept Explanation)\n\n"
                f"Course material ke mutabiq, **{primary_unit.subtopic}** ek core concept hai "
                f"in **{primary_unit.topic}**."
            )
        else:
            explanation_blocks.append(
                f"### 📘 {primary_unit.subtopic}\n\n"
                f"According to the ingested course material for **{primary_unit.topic}**, "
                f"here is the grounded breakdown:"
            )

        explanation_blocks.append(f"\n{primary_unit.content_text}\n")

        # Include Mathematical Formulation if present
        if primary_unit.key_formulas:
            formula_md = "\n**Key Mathematical Formulation:**\n"
            for f in primary_unit.key_formulas:
                formula_md += f"- $${f}$$\n"
            explanation_blocks.append(formula_md)

        # Include Visual Diagram / Figure reference if present
        if primary_unit.diagram:
            fig = primary_unit.diagram
            explanation_blocks.append(
                f"\n> 🖼️ **Associated Visual Material [{primary_unit.location}]:**\n"
                f"> **{fig.figure_id}** — *{fig.caption}*\n"
                f"> Context: {fig.context_text}\n"
            )

        # Cross-Modal Perspectives (linking slides, textbook, video)
        if supporting_units:
            cross_modal_text = "\n### 🔗 Cross-Modal Connections Across Course Assets:\n"
            for supp in supporting_units[:2]:
                icon = "🎥" if supp.source_type == "lecture_video" else ("📖" if supp.source_type == "textbook" else "📊")
                cross_modal_text += (
                    f"- {icon} **{supp.source_title}** (`{supp.location}`):\n"
                    f"  *{supp.content_text[:160]}...*\n"
                )
            explanation_blocks.append(cross_modal_text)

        # Hinglish intuitive wrap-up
        if language_mode == "Hinglish":
            explanation_blocks.append(
                "\n💡 **Intuitive Summary (Hinglish):**\n"
                "Asaan lafzon mein: Yeh concept ensure karta hai ki model data ke patterns ko sahi se generalise "
                "kare bina overfitting ya underfitting ke. Source citations check karke aap directly lecture aur textbook mein jaa sakte hain."
            )

        full_explanation = "\n".join(explanation_blocks)

        return {
            "in_scope": True,
            "confidence": confidence,
            "explanation": full_explanation,
            "citations": citations,
            "primary_topic": primary_unit.topic,
            "primary_subtopic": primary_unit.subtopic,
            "refusal_reason": None
        }
