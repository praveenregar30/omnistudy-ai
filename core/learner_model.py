"""Cognitive Learner Model using Bayesian Knowledge Tracing (BKT) and IRT.

Maintains per-topic mastery estimates that dynamically update after every
quiz question and tutor conversation, with cold-start diagnostic intake and
spaced-repetition memory retention tracking.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class BKTParameters:
    """Standard Bayesian Knowledge Tracing parameters for a knowledge component."""
    p_l0: float = 0.20  # Prior probability of knowing concept initially
    p_t: float = 0.15   # Probability of learning concept during opportunity
    p_g: float = 0.20   # Guess rate (answering correctly despite unlearned)
    p_s: float = 0.10   # Slip rate (answering incorrectly despite learned)

@dataclass
class TopicMastery:
    topic: str
    mastery_prob: float
    interactions_count: int = 0
    correct_count: int = 0
    last_studied_timestamp: float = field(default_factory=time.time)
    stability_days: float = 1.0  # Spaced repetition memory stability (S)

    def retention_probability(self, current_time: Optional[float] = None) -> float:
        """Ebbinghaus forgetting curve: R(t) = exp(-delta_t / S)."""
        now = current_time or time.time()
        elapsed_days = max(0.0, (now - self.last_studied_timestamp) / 86400.0)
        # Avoid division by zero
        decay = elapsed_days / max(0.1, self.stability_days)
        return round(self.mastery_prob * math.exp(-decay), 3)

@dataclass
class StudentProfile:
    student_id: str
    name: str
    education_level: str = "Undergraduate"  # 'Beginner', 'Undergraduate', 'Advanced'
    diagnostic_completed: bool = False
    topic_masteries: Dict[str, TopicMastery] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    question_history_ids: List[str] = field(default_factory=list)

class BayesianLearnerModel:
    """Manages student cognitive state across all curriculum topics."""

    DEFAULT_TOPICS = [
        "Supervised Learning",
        "Classification",
        "Model Evaluation & Regularization",
        "Deep Learning",
        "Unsupervised Learning"
    ]

    def __init__(self, default_params: Optional[BKTParameters] = None):
        self.params = default_params or BKTParameters()

    def create_new_student(self, student_id: str, name: str, level: str = "Undergraduate", *args, **kwargs) -> StudentProfile:
        """Initializes a student profile with cold-start diagnostic defaults."""
        include_default_topics = kwargs.get("include_default_topics", True)
        if args:
            include_default_topics = bool(args[0])
        initial_p0 = 0.15 if level == "Beginner" else (0.45 if level == "Advanced" else 0.25)
        profile = StudentProfile(
            student_id=student_id,
            name=name,
            education_level=level,
            diagnostic_completed=False,
            topic_masteries={},
            interaction_history=[],
            question_history_ids=[]
        )
        if include_default_topics:
            for topic in self.DEFAULT_TOPICS:
                profile.topic_masteries[topic] = TopicMastery(
                    topic=topic,
                    mastery_prob=initial_p0,
                    interactions_count=0,
                    correct_count=0,
                    last_studied_timestamp=time.time(),
                    stability_days=2.0
                )
        return profile

    def apply_diagnostic_result(self, profile: StudentProfile, topic_scores: Dict[str, float]) -> None:
        """Calibrates initial prior probabilities P(L0) after diagnostic intake."""
        for topic, score in topic_scores.items():
            if topic in profile.topic_masteries:
                # Bound between 0.05 and 0.90 to preserve Bayesian responsiveness
                calibrated = min(0.90, max(0.08, score))
                profile.topic_masteries[topic].mastery_prob = round(calibrated, 3)
                profile.topic_masteries[topic].last_studied_timestamp = time.time()
        profile.diagnostic_completed = True
        profile.interaction_history.append({
            "type": "diagnostic_intake",
            "timestamp": time.time(),
            "scores": topic_scores
        })

    def update_after_assessment_item(
        self,
        profile: StudentProfile,
        topic: str,
        is_correct: bool,
        question_id: str,
        question_difficulty: float = 0.5  # IRT difficulty parameter b
    ) -> Dict[str, Any]:
        """Performs Bayesian Knowledge Tracing posterior update and transition."""
        if topic not in profile.topic_masteries:
            profile.topic_masteries[topic] = TopicMastery(
                topic=topic,
                mastery_prob=self.params.p_l0
            )

        tm = profile.topic_masteries[topic]
        p_prev = tm.mastery_prob
        p_g = self.params.p_g
        p_s = self.params.p_s
        p_t = self.params.p_t

        # 1. Posterior calculation conditioned on observed correctness
        if is_correct:
            # P(L_t | Correct) = (P_prev * (1 - P_s)) / (P_prev * (1 - P_s) + (1 - P_prev) * P_g)
            numerator = p_prev * (1.0 - p_s)
            denominator = numerator + ((1.0 - p_prev) * p_g)
        else:
            # P(L_t | Incorrect) = (P_prev * P_s) / (P_prev * P_s + (1 - P_prev) * (1 - P_g))
            numerator = p_prev * p_s
            denominator = numerator + ((1.0 - p_prev) * (1.0 - p_g))

        p_posterior = numerator / max(1e-7, denominator)

        # 2. Transition step (learning during the interaction)
        # P(L_{t+1}) = P_posterior + (1 - P_posterior) * P_t
        p_next = p_posterior + ((1.0 - p_posterior) * p_t)
        p_next = min(0.99, max(0.01, p_next))

        # 3. Update memory stability (Spaced repetition reinforcement)
        if is_correct:
            tm.stability_days = min(30.0, tm.stability_days * 1.5)
            tm.correct_count += 1
        else:
            tm.stability_days = max(1.0, tm.stability_days * 0.7)

        tm.mastery_prob = round(p_next, 3)
        tm.interactions_count += 1
        tm.last_studied_timestamp = time.time()

        # Track question ID to prevent repeated questions across assessments
        if question_id not in profile.question_history_ids:
            profile.question_history_ids.append(question_id)

        update_info = {
            "topic": topic,
            "question_id": question_id,
            "was_correct": is_correct,
            "previous_mastery": round(p_prev, 3),
            "updated_mastery": round(p_next, 3),
            "delta": round(p_next - p_prev, 3),
            "stability_days": round(tm.stability_days, 1),
            "timestamp": time.time()
        }
        profile.interaction_history.append(update_info)
        return update_info

    def update_after_conversation(self, profile: StudentProfile, topic: str, depth: str = "medium") -> float:
        """Rewards conceptual exploration during tutor chat."""
        if topic not in profile.topic_masteries:
            return 0.20
        tm = profile.topic_masteries[topic]
        gain = 0.04 if depth == "light" else (0.08 if depth == "medium" else 0.12)
        tm.mastery_prob = min(0.95, round(tm.mastery_prob + gain, 3))
        tm.last_studied_timestamp = time.time()
        return tm.mastery_prob

    def get_weak_topics(self, profile: StudentProfile, threshold: float = 0.60) -> List[Dict[str, Any]]:
        """Identifies topics requiring urgent remediation."""
        weak = []
        for topic, tm in profile.topic_masteries.items():
            if tm.mastery_prob < threshold:
                weak.append({
                    "topic": topic,
                    "mastery": tm.mastery_prob,
                    "retention": tm.retention_probability(),
                    "gap": round(1.0 - tm.mastery_prob, 3)
                })
        weak.sort(key=lambda x: x["mastery"])
        return weak
