"""OmniStudy AI - Web Application & Interactive Tutor Dashboard.

Track D: Personalized Tutoring & Adaptive Learning
Multimodal AI Hackathon 2026 (KamandPrompt, IIT Mandi, Augli.ai)
"""

import os
import time
import json
import streamlit as st
import pandas as pd
from typing import Dict, Any

from core.ingestion import CourseKnowledgeBase, ContentUnit
from core.learner_model import BayesianLearnerModel, StudentProfile
from core.grounding_engine import GroundingEngine
from core.assessment_engine import AdaptiveAssessmentEngine, AssessmentQuestion
from core.enhancements import EnhancementsManager
from core.evaluation_benchmarks import BenchmarkSuite

# Page Configuration
st.set_page_config(
    page_title="OmniStudy AI | Multimodal Study Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .citation-pill {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .refusal-alert {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
@st.cache_resource
def get_core_instances():
    kb = CourseKnowledgeBase()
    learner = BayesianLearnerModel()
    grounding = GroundingEngine(kb)
    assessment = AdaptiveAssessmentEngine(kb)
    enhancements = EnhancementsManager(kb)
    suite = BenchmarkSuite(kb, grounding, assessment, learner)
    return kb, learner, grounding, assessment, enhancements, suite

kb, learner, grounding, assessment, enhancements, suite = get_core_instances()

if "students" not in st.session_state:
    st.session_state.students = {
        "Priya": learner.create_new_student("std_01", "Priya (Undergraduate)", "Undergraduate"),
        "Aarav": learner.create_new_student("std_02", "Aarav (Beginner)", "Beginner"),
        "Rohan": learner.create_new_student("std_03", "Rohan (Advanced)", "Advanced")
    }
    # Calibrate initial student states
    learner.apply_diagnostic_result(st.session_state.students["Priya"], {
        "Supervised Learning": 0.40,
        "Classification": 0.35,
        "Model Evaluation & Regularization": 0.25,
        "Deep Learning": 0.20,
        "Unsupervised Learning": 0.30
    })

if "current_student_key" not in st.session_state:
    st.session_state.current_student_key = "Priya"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_quiz" not in st.session_state:
    st.session_state.active_quiz = []

if "quiz_attempts" not in st.session_state:
    st.session_state.quiz_attempts = {}

current_student = st.session_state.students[st.session_state.current_student_key]

# ==========================================
# SIDEBAR: Profile, Course State & Ingestion
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.markdown("### 🎓 OmniStudy AI")
    st.caption("Track D: Personalized Tutoring & Adaptive Learning")

    st.markdown("---")
    st.subheader("👤 Learner Profile")
    student_choice = st.selectbox(
        "Active Student:",
        options=list(st.session_state.students.keys()),
        index=list(st.session_state.students.keys()).index(st.session_state.current_student_key)
    )
    if student_choice != st.session_state.current_student_key:
        st.session_state.current_student_key = student_choice
        st.rerun()

    # Calculate overall BKT mastery
    t_masteries = [tm.mastery_prob for tm in current_student.topic_masteries.values()]
    avg_mastery = sum(t_masteries) / max(1, len(t_masteries))
    
    st.metric(
        label="Overall BKT Mastery P(L)",
        value=f"{int(avg_mastery * 100)}%",
        delta=f"{len(current_student.interaction_history)} interactions"
    )

    lang_mode = st.radio("Language Mode:", ["English", "Hinglish (Bilingual)"], horizontal=True)

    st.markdown("---")
    st.subheader("📚 Multimodal Ingestion")
    st.info(f"**Course:** {kb.course_name}\n\n"
            f"• **Ingested Units:** {len(kb.units)}\n"
            f"• **Lectures:** 5 videos (timestamped)\n"
            f"• **Textbooks:** 3 volumes (paginated)\n"
            f"• **Slide Decks:** 3 decks (with diagrams)")

    with st.expander("➕ Ingest Custom Document"):
        custom_title = st.text_input("Title:", "Supplementary Lecture Notes")
        custom_topic = st.selectbox("Topic:", list(kb.topics.keys()))
        custom_source = st.selectbox("Format:", ["lecture_video", "textbook", "slide_deck"])
        custom_loc = st.text_input("Source Location:", "Video @ 02:00 / p. 12 / Slide 4")
        custom_text = st.text_area("Content Text:", "Type or paste transcript/notes here...")
        if st.button("Ingest into Knowledge Base"):
            if custom_text.strip():
                kb.ingest_custom_document(custom_title, custom_text, custom_source, custom_topic, custom_title, custom_loc)
                st.success("Successfully ingested into live Multimodal Knowledge Base!")
                st.rerun()

# ==========================================
# MAIN INTERFACE HEADER
# ==========================================
st.markdown('<div class="main-header">OmniStudy AI: Grounded Multimodal Study Companion</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Source-Grounding (Textbooks, Slides, Videos) • Bayesian Knowledge Tracing • Adaptive Assessment</div>', unsafe_allow_html=True)

# TOP TABS
tab_chat, tab_assessment, tab_learner, tab_graph, tab_revision, tab_eval = st.tabs([
    "💬 Source-Grounded Tutor",
    "📝 Adaptive Assessments",
    "🧠 Learner Cognitive State (BKT)",
    "🗺️ Course Prerequisite DAG",
    "⚡ Revision Kit & Exam Scheduler",
    "📊 Benchmarking & Evaluation (RAGAS)"
])

# ==========================================
# TAB 1: SOURCE-GROUNDED TUTOR CHAT
# ==========================================
with tab_chat:
    st.markdown("#### 💬 Ask Your Grounded AI Tutor")
    st.caption("All answers are strictly verified and cited against lecture videos, textbooks, and slides. Out-of-syllabus queries are declined.")

    # Demo quick-action chips
    st.markdown("**Quick Demo Prompts:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Why 1/2 factor in MSE?"):
            st.session_state.pending_query = "Why is 1/2 factor included in the linear regression cost function?"
    with col2:
        if st.button("Explain Sigmoid in Logistic Reg."):
            st.session_state.pending_query = "What is the formula and output range of the sigmoid activation function in logistic regression?"
    with col3:
        if st.button("L1 Lasso vs L2 Ridge"):
            st.session_state.pending_query = "Why does L1 Lasso regularization produce sparse models with zero weights compared to L2 Ridge?"
    with col4:
        if st.button("⚠️ Test Off-Curriculum"):
            st.session_state.pending_query = "How does quantum key distribution protocol BB84 prevent eavesdropping?"

    query_input = st.text_input(
        "Type your question:",
        value=st.session_state.get("pending_query", ""),
        key="query_box"
    )

    if st.button("Send to Tutor", type="primary"):
        if query_input.strip():
            # Clear pending
            st.session_state.pending_query = ""
            with st.spinner("Retrieving multimodal evidence & synthesizing source-grounded response..."):
                answer = grounding.answer_query(
                    query=query_input,
                    student_level=current_student.education_level,
                    language_mode="Hinglish" if "Hinglish" in lang_mode else "English"
                )

                # Update learner model if in-scope
                if answer["in_scope"] and answer["primary_topic"]:
                    learner.update_after_conversation(current_student, answer["primary_topic"], depth="medium")

                st.session_state.chat_history.append({
                    "query": query_input,
                    "answer": answer,
                    "timestamp": time.time()
                })

    # Render Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        for chat in reversed(st.session_state.chat_history):
            ans = chat["answer"]
            st.markdown(f"**🧑 Student:** {chat['query']}")
            
            if not ans["in_scope"]:
                st.markdown(f"""
                <div class="refusal-alert">
                    <strong>⚠️ Out-of-Curriculum Refusal:</strong><br>
                    {ans['explanation']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(ans["explanation"])
                
                # Render Citation Cards
                st.markdown("##### 📚 Exact Source Citations:")
                c_cols = st.columns(len(ans["citations"][:3]))
                for idx, c in enumerate(ans["citations"][:3]):
                    with c_cols[idx]:
                        icon = "🎥" if c["source_type"] == "lecture_video" else ("📖" if c["source_type"] == "textbook" else "📊")
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{icon} {c['source_title']}</strong><br>
                            <span class="citation-pill">📍 {c['location']}</span><br>
                            <small><em>"{c['excerpt']}"</em></small>
                            {"<br><small>🖼️ <b>Figure:</b> " + c['diagram'] + "</small>" if c['diagram'] else ""}
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown("---")

# ==========================================
# TAB 2: ADAPTIVE ASSESSMENTS
# ==========================================
with tab_assessment:
    st.markdown("#### 📝 Adaptive Assessments & Diagnostic Quizzes")
    st.caption("Questions calibrated to student BKT mastery with anti-repetition filtering and misconception diagnostics.")

    q_col1, q_col2, q_col3 = st.columns([2, 1, 1])
    with q_col1:
        target_t = st.selectbox("Target Curriculum Scope:", ["All Topics"] + list(kb.topics.keys()))
    with q_col2:
        num_q = st.slider("Number of Questions:", min_value=1, max_value=5, value=3)
    with q_col3:
        st.write("")
        st.write("")
        if st.button("Generate Adaptive Quiz", type="primary"):
            topic_filter = None if target_t == "All Topics" else target_t
            quiz = assessment.select_adaptive_quiz(current_student, num_questions=num_q, target_topic=topic_filter)
            st.session_state.active_quiz = quiz
            st.session_state.quiz_attempts = {}

    if st.session_state.active_quiz:
        st.markdown(f"### 📋 Active Quiz ({len(st.session_state.active_quiz)} Questions)")
        st.caption("🔒 Verified by Dual-Pass Validator • 🚫 Zero Repetition Filter: ACTIVE")

        for idx, q in enumerate(st.session_state.active_quiz):
            st.markdown(f"#### Q{idx+1}: {q.question_text}")
            st.markdown(
                f"<span class='citation-pill'>Topic: {q.topic}</span> "
                f"<span class='citation-pill'>Difficulty: {q.difficulty_label} (b={q.difficulty})</span> "
                f"<span class='citation-pill'>Location: {q.source_location}</span>",
                unsafe_allow_html=True
            )

            user_ans_idx = st.radio(
                f"Select your answer for Q{idx+1}:",
                options=list(range(len(q.options))),
                format_func=lambda i: q.options[i],
                key=f"q_choice_{q.question_id}"
            )

            if st.button(f"Submit Answer for Q{idx+1}", key=f"btn_sub_{q.question_id}"):
                attempt = assessment.grade_attempt(q, user_ans_idx)
                st.session_state.quiz_attempts[q.question_id] = attempt
                
                # Update Learner BKT State
                bkt_info = learner.update_after_assessment_item(
                    profile=current_student,
                    topic=q.topic,
                    is_correct=attempt.is_correct,
                    question_id=q.question_id,
                    question_difficulty=q.difficulty
                )
                st.session_state[f"bkt_{q.question_id}"] = bkt_info

            # Display feedback if attempted
            if q.question_id in st.session_state.quiz_attempts:
                att = st.session_state.quiz_attempts[q.question_id]
                bkt_inf = st.session_state.get(f"bkt_{q.question_id}")
                st.markdown(att.cited_feedback)
                if bkt_inf:
                    delta_symbol = "+" if bkt_inf["delta"] >= 0 else ""
                    st.info(
                        f"📊 **BKT Knowledge Update:** Topic **{bkt_inf['topic']}** mastery changed from "
                        f"**{bkt_inf['previous_mastery']:.2f}** ➔ **{bkt_inf['updated_mastery']:.2f}** "
                        f"({delta_symbol}{bkt_inf['delta']:.2f})"
                    )
            st.markdown("---")

# ==========================================
# TAB 3: LEARNER COGNITIVE STATE (BKT)
# ==========================================
with tab_learner:
    st.markdown("#### 🧠 Cognitive Learner Model: Bayesian Knowledge Tracing")
    st.caption("Dynamic mastery estimation P(L), IRT item calibrations, and Ebbinghaus retention decay.")

    st.markdown(f"**Student:** `{current_student.name}` | **Level:** `{current_student.education_level}`")

    # Topic Mastery Table & Gauges
    rows = []
    for topic, tm in current_student.topic_masteries.items():
        retention = tm.retention_probability()
        status = "🟢 Mastered" if tm.mastery_prob >= 0.75 else ("🟡 Developing" if tm.mastery_prob >= 0.45 else "🔴 Remediation Needed")
        rows.append({
            "Topic": topic,
            "Mastery P(L)": f"{tm.mastery_prob:.2f}",
            "Retention R(t)": f"{retention:.2f}",
            "Interactions": tm.interactions_count,
            "Correct Answers": tm.correct_count,
            "Memory Stability": f"{tm.stability_days:.1f} days",
            "Cognitive Status": status
        })
    df_mastery = pd.DataFrame(rows)
    st.dataframe(df_mastery, use_container_width=True)

    # Diagnostic Intake Button
    with st.expander("🛠️ Re-Run Cold-Start Diagnostic Intake"):
        st.write("Calibrate prior knowledge P(L0) for this student profile.")
        diag_cols = st.columns(len(current_student.topic_masteries))
        new_scores = {}
        for idx, (t, tm) in enumerate(current_student.topic_masteries.items()):
            with diag_cols[idx]:
                new_scores[t] = st.slider(f"{t[:15]}...", 0.0, 1.0, tm.mastery_prob, 0.05)
        if st.button("Apply Diagnostic Calibration"):
            learner.apply_diagnostic_result(current_student, new_scores)
            st.success("Prior probabilities successfully updated!")
            st.rerun()

# ==========================================
# TAB 4: PREREQUISITE DAG FLOW
# ==========================================
with tab_graph:
    st.markdown("#### 🗺️ Visual Course Flow & Prerequisite Dependency DAG")
    st.caption("Visual representation of curriculum concepts and prerequisites. Nodes are dynamically colored by current student mastery.")

    mermaid_code = enhancements.generate_course_flow_mermaid(current_student)
    st.markdown(f"```mermaid\n{mermaid_code}\n```")

    st.markdown("##### 📌 Prerequisite Graph Legend:")
    leg_col1, leg_col2, leg_col3 = st.columns(3)
    with leg_col1:
        st.markdown(r"🟢 **Green Node:** Mastered Concept ($P(L) \ge 0.75$)")
    with leg_col2:
        st.markdown(r"🟡 **Yellow Node:** Developing Concept ($0.45 \le P(L) < 0.75$)")
    with leg_col3:
        st.markdown(r"🔴 **Red Node:** Urgent Remediation Needed ($P(L) < 0.45$)")

# ==========================================
# TAB 5: REVISION KIT & EXAM SCHEDULER
# ==========================================
with tab_revision:
    st.markdown("#### ⚡ High-Yield Revision Kit & Spaced Repetition Exam Scheduler")
    st.caption("Custom study materials generated targeting the student's specific cognitive gaps.")

    rev_kit = enhancements.generate_revision_kit(current_student)

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("### 📇 Smart Flashcards for Weak Concepts")
        for fc in rev_kit["flashcards"][:3]:
            with st.expander(f"📌 {fc['topic']}: {fc['subtopic']}"):
                st.markdown(f"**Question:** {fc['front']}")
                st.markdown(f"**Answer:** {fc['back']}")
                if fc['formula'] != "N/A":
                    st.latex(fc['formula'])
                st.caption(f"Source: {fc['source_cite']}")

    with r_col2:
        st.markdown("### 🎙️ 2-Minute Audio Brief Script")
        st.markdown("Listen to or read your personalized audio summary:")
        st.info(rev_kit["audio_brief_script"])
        st.caption("🔊 Integrated Web Audio Synthesizer ready for voice tutoring playback.")

    st.markdown("---")
    st.markdown("### 📅 Adaptive Exam Study Calendar (Ebbinghaus Forgetting Curve)")
    days_left = st.slider("Days Until Final Exam:", min_value=3, max_value=30, value=7)
    schedule = enhancements.generate_exam_study_schedule(current_student, days_until_exam=days_left)

    sched_rows = []
    for s in schedule:
        sched_rows.append({
            "Day": s["label"],
            "Focus Topics": ", ".join(s["focus_topics"]),
            "Hours": f"{s['allocated_hours']} hrs",
            "Recommended Activity": s["recommended_activity"]
        })
    st.dataframe(pd.DataFrame(sched_rows), use_container_width=True)

# ==========================================
# TAB 6: BENCHMARKING & EVALUATION (RAGAS)
# ==========================================
with tab_eval:
    st.markdown("#### 📊 System Evaluation & Rigorous Benchmarks")
    st.caption("Quantitative validation using RAGAS-style metrics and multi-session student cohort simulations.")

    if os.path.exists("benchmark_results.png"):
        st.image("benchmark_results.png", caption="System Evaluation Metrics (RAGAS) & Simulated Student BKT Mastery Gain Curves", use_container_width=True)

    if os.path.exists("benchmark_report.json"):
        with open("benchmark_report.json") as f:
            bench_data = json.load(f)

        rag_m = bench_data["rag_metrics"]
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Faithfulness", f"{rag_m['faithfulness']:.2f}")
        m_col2.metric("Answer Relevancy", f"{rag_m['answer_relevancy']:.2f}")
        m_col3.metric("Context Recall", f"{rag_m['context_recall']:.2f}")
        m_col4.metric("Refusal Accuracy", f"{rag_m['refusal_accuracy']:.2f}")
        m_col5.metric("Question Repetition", f"{bench_data['question_repetition_rate']:.1f}%")

    if st.button("Re-Run Live Benchmark Suite", type="primary"):
        with st.spinner("Executing Golden Test Set benchmarks and multi-session cohort simulations..."):
            suite.generate_benchmark_plot("benchmark_results.png")
            st.success("Benchmarks successfully re-computed and visual charts updated!")
            st.rerun()

st.markdown("---")
st.caption("OmniStudy AI • Developed for Multimodal AI Hackathon 2026 • KamandPrompt (IIT Mandi), Augli.ai, PurpleRain Tech, LDV Labs")
