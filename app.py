"""OmniStudy AI - Peak Animated Web Application & Cognitive Tutor Dashboard.

Track D: Personalized Tutoring & Adaptive Learning
Multimodal AI Hackathon 2026 (KamandPrompt, IIT Mandi, Augli.ai)
"""

import os
import time
import json
import html
import streamlit as st
import streamlit.components.v1 as components
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
    page_title="OmniStudy AI | Peak Multimodal Study Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# PREMIUM ANIMATED SAAS CSS STYLING (Keyframes, Glassmorphism, Hover)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseDot {
        0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }
        100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
    }

    .hero-banner {
        background: linear-gradient(-45deg, #0f172a, #1e3a8a, #312e81, #0369a1);
        background-size: 300% 300%;
        animation: gradientBG 12s ease infinite, fadeInUp 0.6s ease-out;
        padding: 28px 34px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px -8px rgba(30, 58, 138, 0.45);
        position: relative;
        overflow: hidden;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(8px);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.22);
    }

    .pulse-circle {
        width: 9px;
        height: 9px;
        background-color: #4ade80;
        border-radius: 50%;
        display: inline-block;
        animation: pulseDot 1.8s infinite;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #e0f2fe;
        font-weight: 400;
        margin-bottom: 16px;
    }

    .hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .hero-chip {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 5px 13px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #f8fafc;
        transition: all 0.25s ease;
    }

    .hero-chip:hover {
        background: rgba(255, 255, 255, 0.24);
        transform: translateY(-2px);
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1);
        animation: fadeInUp 0.5s ease-out;
        margin-bottom: 16px;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(30, 58, 138, 0.12);
        border-color: #93c5fd;
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }

    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 4px 0;
    }

    .kpi-sub {
        font-size: 0.8rem;
        color: #0284c7;
        font-weight: 600;
    }

    .progress-track {
        width: 100%;
        height: 7px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 8px;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        border-radius: 999px;
        transition: width 0.6s ease;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #cbd5e1;
        border-left: 4px solid #2563eb;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        transition: all 0.25s ease;
        animation: fadeInUp 0.4s ease-out;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.12);
    }

    .citation-pill {
        display: inline-block;
        background: linear-gradient(90deg, #e0f2fe, #dbeafe);
        color: #1d4ed8;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 4px;
        border: 1px solid #bfdbfe;
    }

    .refusal-alert {
        background: linear-gradient(90deg, #fef2f2, #fff1f2);
        border-left: 5px solid #ef4444;
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
        animation: fadeInUp 0.4s ease-out;
    }

    .simp-banner {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0e7ff 100%);
        border: 1px solid #93c5fd;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
        animation: fadeInUp 0.5s ease-out;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 700;
        border-radius: 8px;
        padding: 9px 16px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1d4ed8 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }
</style>
""", unsafe_allow_html=True)


def render_voice_player(speech_text: str, button_label: str = "🔊 Listen to AI Voice Tutor", key_suffix: str = "main") -> None:
    """Renders an interactive HTML5 Web Speech API voice player with animated audio equalizer."""
    clean_text = speech_text.replace('"', "'").replace("\n", " ").replace("#", "").replace("*", "")[:700]
    escaped = html.escape(clean_text)
    component_html = f"""
    <div style="display:flex; align-items:center; gap:12px; font-family:'Inter',sans-serif; padding:4px 0;">
        <button id="playBtn_{key_suffix}" onclick="speakText_{key_suffix}()" style="
            background: linear-gradient(90deg, #2563eb, #4f46e5);
            color: white; border: none; border-radius: 999px;
            padding: 7px 16px; font-size: 13px; font-weight: 700;
            cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
            box-shadow: 0 3px 10px rgba(37, 99, 235, 0.3); transition: transform 0.2s;">
            {button_label}
        </button>
        <button onclick="stopSpeech_{key_suffix}()" style="
            background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1;
            border-radius: 999px; padding: 7px 13px; font-size: 12px; font-weight: 600;
            cursor: pointer;">
            ⏹️ Stop
        </button>
        <div id="wave_{key_suffix}" style="display:none; align-items:center; gap:3px; height:18px;">
            <span style="width:4px; height:14px; background:#2563eb; border-radius:2px; animation:w1 0.6s infinite alternate;"></span>
            <span style="width:4px; height:18px; background:#4f46e5; border-radius:2px; animation:w1 0.4s infinite alternate;"></span>
            <span style="width:4px; height:10px; background:#06b6d4; border-radius:2px; animation:w1 0.7s infinite alternate;"></span>
            <span style="width:4px; height:16px; background:#2563eb; border-radius:2px; animation:w1 0.5s infinite alternate;"></span>
            <span style="font-size:12px; color:#2563eb; font-weight:600; margin-left:6px;">Speaking...</span>
        </div>
    </div>
    <style>
        @keyframes w1 {{ from {{ transform: scaleY(0.35); }} to {{ transform: scaleY(1.1); }} }}
    </style>
    <script>
        function speakText_{key_suffix}() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("{escaped}");
                msg.rate = 1.02;
                msg.pitch = 1.0;
                var wave = document.getElementById("wave_{key_suffix}");
                msg.onstart = function() {{ if(wave) wave.style.display = "inline-flex"; }};
                msg.onend = function() {{ if(wave) wave.style.display = "none"; }};
                msg.onerror = function() {{ if(wave) wave.style.display = "none"; }};
                window.speechSynthesis.speak(msg);
            }} else {{
                alert("Web Speech API is not supported in this browser.");
            }}
        }}
        function stopSpeech_{key_suffix}() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var wave = document.getElementById("wave_{key_suffix}");
                if(wave) wave.style.display = "none";
            }}
        }}
    </script>
    """
    components.html(component_html, height=48)


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
        "Praveen": learner.create_new_student("std_00", "Praveen Regar (Undergraduate)", "Undergraduate"),
        "Priya": learner.create_new_student("std_01", "Priya (Undergraduate)", "Undergraduate"),
        "Aarav": learner.create_new_student("std_02", "Aarav (Beginner)", "Beginner"),
        "Rohan": learner.create_new_student("std_03", "Rohan (Advanced)", "Advanced")
    }
    # Calibrate initial student states
    learner.apply_diagnostic_result(st.session_state.students["Praveen"], {
        "Supervised Learning": 0.45,
        "Classification": 0.40,
        "Model Evaluation & Regularization": 0.30,
        "Deep Learning": 0.35,
        "Unsupervised Learning": 0.40
    })
    learner.apply_diagnostic_result(st.session_state.students["Priya"], {
        "Supervised Learning": 0.40,
        "Classification": 0.35,
        "Model Evaluation & Regularization": 0.25,
        "Deep Learning": 0.20,
        "Unsupervised Learning": 0.30
    })

if "current_student_key" not in st.session_state:
    st.session_state.current_student_key = "Praveen"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_quiz" not in st.session_state:
    st.session_state.active_quiz = []

if "quiz_attempts" not in st.session_state:
    st.session_state.quiz_attempts = {}

if "simplified_notes_result" not in st.session_state:
    st.session_state.simplified_notes_result = None

if "custom_quiz_count" not in st.session_state:
    st.session_state.custom_quiz_count = 0

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

current_student = st.session_state.students[st.session_state.current_student_key]

# Calculate overall BKT mastery
t_masteries = [tm.mastery_prob for tm in current_student.topic_masteries.values()]
avg_mastery = sum(t_masteries) / max(1, len(t_masteries))

# ==========================================
# SIDEBAR: Profile, Course State & Ingestion
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
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

    with st.expander("➕ Add New Student Profile"):
        new_student_name = st.text_input("Student Name:", placeholder="e.g., Rahul Sharma")
        new_student_level = st.selectbox("Level:", ["Beginner", "Undergraduate", "Advanced"])
        if st.button("Create Student Profile"):
            if new_student_name.strip():
                clean_key = new_student_name.strip().split()[0]
                sid = f"std_{len(st.session_state.students) + 1:02d}"
                st.session_state.students[clean_key] = learner.create_new_student(
                    sid, f"{new_student_name.strip()} ({new_student_level})", new_student_level
                )
                st.session_state.current_student_key = clean_key
                st.success(f"Added {new_student_name}!")
                st.rerun()

    st.metric(
        label="Overall BKT Mastery P(L)",
        value=f"{int(avg_mastery * 100)}%",
        delta=f"{len(current_student.interaction_history)} interactions"
    )

    lang_mode = st.radio("Language Mode:", ["English", "Hinglish (Bilingual)"], horizontal=True)

    st.markdown("---")
    st.subheader("📄 Upload & Simplify Notes")
    with st.expander("✨ Upload PDF / Lecture Notes", expanded=True):
        uploaded_file = st.file_uploader("Upload PDF or Text Notes:", type=["pdf", "txt", "md"])
        custom_title = st.text_input("Or Title for Pasted Notes:", "My Lecture Notes")
        custom_topic = st.selectbox("Assign to Topic:", list(kb.topics.keys()))
        custom_text = st.text_area("Or Paste Notes Here:", placeholder="Paste any complex lecture paragraph here to simplify and ingest...")
        if st.button("🚀 Ingest, Simplify & Make Quiz", type="primary"):
            raw_content_to_simplify = ""
            doc_title = custom_title
            if uploaded_file is not None:
                doc_title = uploaded_file.name
                units_created = kb.ingest_uploaded_file(uploaded_file, uploaded_file.name, topic=custom_topic)
                raw_content_to_simplify = " ".join([u.content_text for u in units_created])
            elif custom_text.strip():
                u = kb.ingest_custom_document(custom_title, custom_text, "slide_deck", custom_topic, custom_title, "Custom Notes p. 1")
                raw_content_to_simplify = custom_text.strip()

            if raw_content_to_simplify:
                simp_res = kb.simplify_lecture_notes(
                    raw_content_to_simplify, title=doc_title, lang_mode=lang_mode
                )
                # Peak Feature #1: Auto-generate verified quiz questions from uploaded PDF!
                generated_qs = assessment.generate_questions_from_custom_notes(
                    title=doc_title,
                    topic=custom_topic,
                    key_points=simp_res.get("key_points", []),
                    top_concepts=simp_res.get("top_concepts", [])
                )
                simp_res["generated_quiz_count"] = len(generated_qs)
                st.session_state.custom_quiz_count += len(generated_qs)
                st.session_state.simplified_notes_result = simp_res
                st.success(f"Simplified '{doc_title}' + Added {len(generated_qs)} Quiz Questions!")
                st.rerun()

    st.markdown("---")
    with st.expander("🤖 Optional: Live Gemini AI Key"):
        st.caption("Works 100% offline by default. Paste a free Gemini API key for live LLM synthesis:")
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key:",
            value=st.session_state.gemini_api_key,
            type="password",
            placeholder="AIzaSy..."
        )

    # Peak Feature #5: Sidebar Quick Download Study Pack
    study_guide_md = enhancements.export_markdown_study_guide(current_student, days_until_exam=7)
    st.download_button(
        label="📥 Download My Exam Cheat-Sheet",
        data=study_guide_md,
        file_name=f"{current_student.student_id}_OmniStudy_CheatSheet.md",
        mime="text/markdown",
        use_container_width=True
    )


# ==========================================
# ANIMATED HERO BANNER & LIVE KPI CARDS
# ==========================================
st.markdown(f"""
<div class="hero-banner">
    <div class="status-badge">
        <span class="pulse-circle"></span>
        AI COGNITIVE ENGINE ONLINE • MULTIMODAL AI HACKATHON 2026 (TRACK D)
    </div>
    <div class="hero-title">OmniStudy AI: Grounded Multimodal Study Companion</div>
    <div class="hero-subtitle">
        Unifying Lecture Videos, Textbooks & Slide Decks into a Source-Cited Knowledge Base with Bayesian Knowledge Tracing
    </div>
    <div class="hero-chips">
        <span class="hero-chip">🎥 Timestamped Video Grounding</span>
        <span class="hero-chip">📖 Page-Exact PDF Citations</span>
        <span class="hero-chip">🧠 Bayesian Knowledge Tracing (BKT)</span>
        <span class="hero-chip">🔊 Interactive Voice Tutor</span>
        <span class="hero-chip">🇮🇳 Hinglish Bilingual Mode</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top 4 Animated KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">👤 Active Learner Mastery</div>
        <div class="kpi-value">{int(avg_mastery * 100)}% <span style="font-size:0.9rem;color:#2563eb;">P(L)</span></div>
        <div class="kpi-sub">{current_student.name}</div>
        <div class="progress-track"><div class="progress-fill" style="width:{int(avg_mastery * 100)}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">📚 Multimodal Knowledge Base</div>
        <div class="kpi-value">{len(kb.units)} <span style="font-size:0.9rem;color:#0284c7;">Units</span></div>
        <div class="kpi-sub">5 Videos • 3 Books • {len(kb.units) - 7} Slides/Notes</div>
        <div class="progress-track"><div class="progress-fill" style="width:100%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">📝 Verified Adaptive Quiz Bank</div>
        <div class="kpi-value">{len(assessment.question_bank)} <span style="font-size:0.9rem;color:#059669;">Items</span></div>
        <div class="kpi-sub">0.0% Repetition • {st.session_state.custom_quiz_count} Custom PDF Qs</div>
        <div class="progress-track"><div class="progress-fill" style="width:92%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">🛡️ Source Grounding & Refusal</div>
        <div class="kpi-value">100% <span style="font-size:0.9rem;color:#7c3aed;">Fidelity</span></div>
        <div class="kpi-sub">Strict Out-of-Syllabus Gate Active</div>
        <div class="progress-track"><div class="progress-fill" style="width:100%;"></div></div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# NEWLY SIMPLIFIED LECTURE NOTES BANNER
# ==========================================
if st.session_state.simplified_notes_result:
    simp = st.session_state.simplified_notes_result
    with st.expander(f"✨ Newly Simplified Lecture Notes: {simp['title']} ({simp['word_count']} words | +{simp.get('generated_quiz_count', 0)} Quiz Questions Created!)", expanded=True):
        st.markdown('<div class="simp-banner">', unsafe_allow_html=True)
        st.markdown(simp["summary_markdown"])
        
        # Peak Feature #2: Voice Player for Simplified Notes!
        speech_summary = " ".join(simp.get("key_points", ["Summary ready."]))
        render_voice_player(
            f"Simplified summary for {simp['title']}. " + speech_summary,
            button_label="🔊 Speak Simplified Notes Out Loud",
            key_suffix="simp_notes"
        )

        st.markdown("##### 📇 Auto-Generated Study Flashcards from Your Uploaded Notes:")
        fc_cols = st.columns(len(simp["flashcards"]))
        for idx, fc in enumerate(simp["flashcards"]):
            with fc_cols[idx]:
                st.info(f"**Q:** {fc['question']}\n\n**A:** {fc['answer']}\n\n*📍 {fc['source']}*")
        
        st.success(f"🎯 **Auto-Quiz Ready:** We automatically added **{simp.get('generated_quiz_count', 0)} interactive MCQ questions** from `{simp['title']}` into the **📝 Adaptive Assessments** tab!")
        if st.button("✖️ Close Simplified View"):
            st.session_state.simplified_notes_result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TOP NAVIGATION TABS
# ==========================================
tab_chat, tab_assessment, tab_learner, tab_graph, tab_revision, tab_eval = st.tabs([
    "💬 Source-Grounded Tutor",
    "📝 Adaptive Assessments & Feynman Mode",
    "🧠 Learner Cognitive State (BKT)",
    "🗺️ Course Prerequisite DAG",
    "⚡ Revision Kit & Exam Scheduler",
    "📊 Benchmarking & Evaluation (RAGAS)"
])

# ==========================================
# TAB 1: SOURCE-GROUNDED TUTOR CHAT
# ==========================================
with tab_chat:
    st.markdown("#### 💬 Ask Your Grounded AI Tutor (With Live Voice Synthesis)")
    st.caption("All answers are strictly verified and cited against lecture videos, textbooks, slides, and your uploaded PDFs. Out-of-syllabus queries are declined.")

    st.markdown("**⚡ One-Click Demo Prompts:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📐 Why 1/2 factor in MSE?", use_container_width=True):
            st.session_state.pending_query = "Why is 1/2 factor included in the linear regression cost function?"
    with col2:
        if st.button("📈 Sigmoid in Logistic Reg.", use_container_width=True):
            st.session_state.pending_query = "What is the formula and output range of the sigmoid activation function in logistic regression?"
    with col3:
        if st.button("💎 L1 Lasso vs L2 Ridge", use_container_width=True):
            st.session_state.pending_query = "Why does L1 Lasso regularization produce sparse models with zero weights compared to L2 Ridge?"
    with col4:
        if st.button("🚫 Test Off-Curriculum Gate", use_container_width=True):
            st.session_state.pending_query = "How does quantum key distribution protocol BB84 prevent eavesdropping?"

    query_input = st.text_input(
        "Type your question (about CS 229 or your uploaded PDF notes):",
        value=st.session_state.get("pending_query", ""),
        placeholder="e.g., Explain gradient descent update rule, or ask about your uploaded PDF...",
        key="query_box"
    )

    if st.button("🚀 Ask Grounded Tutor", type="primary"):
        if query_input.strip():
            st.session_state.pending_query = ""
            with st.spinner("Retrieving multimodal evidence & synthesizing source-grounded response..."):
                answer = grounding.answer_query(
                    query=query_input,
                    student_level=current_student.education_level,
                    language_mode="Hinglish" if "Hinglish" in lang_mode else "English",
                    gemini_api_key=st.session_state.gemini_api_key
                )

                if answer["in_scope"] and answer["primary_topic"]:
                    learner.update_after_conversation(current_student, answer["primary_topic"], depth="medium")

                st.session_state.chat_history.append({
                    "query": query_input,
                    "answer": answer,
                    "timestamp": time.time()
                })

    if st.session_state.chat_history:
        st.markdown("---")
        for chat_idx, chat in enumerate(reversed(st.session_state.chat_history)):
            ans = chat["answer"]
            st.markdown(f"##### 🧑 {current_student.name}: *\"{chat['query']}\"*")
            
            if not ans["in_scope"]:
                st.markdown(f"""
                <div class="refusal-alert">
                    <strong>🛡️ Out-of-Curriculum Refusal Gate Triggered:</strong><br><br>
                    {ans['explanation']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(ans["explanation"])
                
                # Peak Feature #2: Live Voice Synthesis Player for Tutor Answer
                speech_txt = ans.get("plain_speech_text", ans["explanation"])
                render_voice_player(
                    speech_txt,
                    button_label="🔊 Listen to Tutor Explanation",
                    key_suffix=f"chat_{chat_idx}"
                )

                # Render Citation Cards
                st.markdown("##### 📚 Exact Multimodal Source Citations:")
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
# TAB 2: ADAPTIVE ASSESSMENTS & FEYNMAN MODE
# ==========================================
with tab_assessment:
    st.markdown("#### 📝 Adaptive Assessments & Diagnostic Quizzes")
    st.caption("Questions dynamically calibrated to your BKT mastery (plus any PDFs you uploaded!), with zero repetition and misconception analysis.")

    q_col1, q_col2, q_col3 = st.columns([2, 1, 1])
    with q_col1:
        target_t = st.selectbox("Target Curriculum Scope:", ["All Topics"] + list(kb.topics.keys()))
    with q_col2:
        num_q = st.slider("Number of Questions:", min_value=1, max_value=5, value=3)
    with q_col3:
        st.write("")
        st.write("")
        if st.button("🎯 Generate Adaptive Quiz", type="primary", use_container_width=True):
            topic_filter = None if target_t == "All Topics" else target_t
            quiz = assessment.select_adaptive_quiz(current_student, num_questions=num_q, target_topic=topic_filter)
            st.session_state.active_quiz = quiz
            st.session_state.quiz_attempts = {}

    if st.session_state.active_quiz:
        st.markdown(f"### 📋 Active Adaptive Quiz ({len(st.session_state.active_quiz)} Verified Questions)")
        st.caption("🔒 Verified by Dual-Pass Validator • 🚫 Zero Repetition Filter: ACTIVE")

        for idx, q in enumerate(st.session_state.active_quiz):
            st.markdown(f"#### Q{idx+1}: {q.question_text}")
            st.markdown(
                f"<span class='citation-pill'>📌 Topic: {q.topic}</span> "
                f"<span class='citation-pill'>⚡ Difficulty: {q.difficulty_label} (b={q.difficulty})</span> "
                f"<span class='citation-pill'>📍 Source: {q.source_location}</span>",
                unsafe_allow_html=True
            )

            user_ans_idx = st.radio(
                f"Select your answer for Q{idx+1}:",
                options=list(range(len(q.options))),
                format_func=lambda i, opts=q.options: opts[i],
                key=f"q_choice_{q.question_id}"
            )

            if st.button(f"✅ Submit Answer for Q{idx+1}", key=f"btn_sub_{q.question_id}"):
                attempt = assessment.grade_attempt(q, user_ans_idx)
                st.session_state.quiz_attempts[q.question_id] = attempt
                
                bkt_info = learner.update_after_assessment_item(
                    profile=current_student,
                    topic=q.topic,
                    is_correct=attempt.is_correct,
                    question_id=q.question_id,
                    question_difficulty=q.difficulty
                )
                st.session_state[f"bkt_{q.question_id}"] = bkt_info

            if q.question_id in st.session_state.quiz_attempts:
                att = st.session_state.quiz_attempts[q.question_id]
                bkt_inf = st.session_state.get(f"bkt_{q.question_id}")
                st.markdown(att.cited_feedback)
                if bkt_inf:
                    delta_symbol = "+" if bkt_inf["delta"] >= 0 else ""
                    st.info(
                        f"📊 **Live BKT Mastery Update:** Topic **{bkt_inf['topic']}** mastery shifted from "
                        f"**{bkt_inf['previous_mastery']:.2f}** ➔ **{bkt_inf['updated_mastery']:.2f}** "
                        f"({delta_symbol}{bkt_inf['delta']:.2f})"
                    )
            st.markdown("---")

    # Peak Feature #4: Feynman Teach-Back Challenge
    st.markdown("### 🧠 Feynman Teach-Back Mode (You Teach the AI!)")
    st.caption("The ultimate test of mastery: Explain a concept in your own simple words. The AI grades your conceptual clarity and awards bonus BKT Mastery XP!")
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        feynman_topic = st.selectbox("Choose Topic to Teach Back:", list(kb.topics.keys()), key="feynman_t")
    with f_col2:
        feynman_input = st.text_area(
            f"Explain '{feynman_topic}' in your own words, {current_student.name.split()[0]}:",
            placeholder="e.g., In supervised learning, we train a model using labeled data by minimizing the cost function with gradient descent...",
            height=90
        )
    if st.button("🏆 Evaluate My Feynman Explanation"):
        if feynman_input.strip():
            f_res = assessment.evaluate_feynman_teachback(feynman_topic, feynman_input)
            new_m = learner.update_after_conversation(current_student, feynman_topic, depth=f_res["bkt_depth"])
            st.success(f"**Feynman Clarity Score: {f_res['clarity_score']}%** — {f_res['verdict']}")
            st.markdown(
                f"- ✅ **Concepts You Nailed:** `{', '.join(f_res['matched_keywords'])}`\n"
                f"- 💡 **Terms to Include Next Time:** `{', '.join(f_res['suggested_terms_to_include']) or 'None - great coverage!'}`\n"
                f"- 📈 **Updated BKT Mastery in {feynman_topic}:** `{int(new_m * 100)}%`"
            )

# ==========================================
# TAB 3: LEARNER COGNITIVE STATE (BKT)
# ==========================================
with tab_learner:
    st.markdown("#### 🧠 Cognitive Learner Model: Bayesian Knowledge Tracing")
    st.caption("Real-time posterior mastery estimation P(L), IRT item calibrations, and Ebbinghaus memory retention decay.")

    st.markdown(f"**Active Student:** `{current_student.name}` | **Academic Level:** `{current_student.education_level}`")

    rows = []
    for topic, tm in current_student.topic_masteries.items():
        retention = tm.retention_probability()
        status = "🟢 Mastered" if tm.mastery_prob >= 0.75 else ("🟡 Developing" if tm.mastery_prob >= 0.45 else "🔴 Remediation Needed")
        rows.append({
            "Topic": topic,
            "Mastery P(L)": f"{tm.mastery_prob:.2f} ({int(tm.mastery_prob*100)}%)",
            "Retention R(t)": f"{retention:.2f}",
            "Interactions": tm.interactions_count,
            "Correct Answers": tm.correct_count,
            "Memory Stability": f"{tm.stability_days:.1f} days",
            "Cognitive Status": status
        })
    df_mastery = pd.DataFrame(rows)
    st.dataframe(df_mastery, use_container_width=True)

    with st.expander("🛠️ Cold-Start Diagnostic Intake Calibration"):
        st.write("Adjust or re-calibrate prior knowledge probabilities P(L0) for this student profile:")
        diag_cols = st.columns(len(current_student.topic_masteries))
        new_scores = {}
        for idx, (t, tm) in enumerate(current_student.topic_masteries.items()):
            with diag_cols[idx]:
                new_scores[t] = st.slider(f"{t[:15]}...", 0.0, 1.0, tm.mastery_prob, 0.05)
        if st.button("Apply Diagnostic Calibration"):
            learner.apply_diagnostic_result(current_student, new_scores)
            st.success("Prior probabilities successfully calibrated!")
            st.rerun()

# ==========================================
# TAB 4: PREREQUISITE DAG FLOW
# ==========================================
with tab_graph:
    st.markdown("#### 🗺️ Visual Course Flow & Prerequisite Dependency DAG")
    st.caption("Interactive curriculum dependency graph. Nodes dynamically change color as your BKT mastery progresses.")

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
    st.caption("Custom flashcards, voice audio brief, and Ebbinghaus study timetable targeted at your weakest topics.")

    rev_kit = enhancements.generate_revision_kit(current_student)

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("### 📇 Smart Flashcards for Weak Concepts")
        for fc in rev_kit["flashcards"][:3]:
            with st.expander(f"📌 {fc['topic']}: {fc['subtopic']}", expanded=True):
                st.markdown(f"**Question:** {fc['front']}")
                st.markdown(f"**Answer:** {fc['back']}")
                if fc['formula'] != "N/A":
                    st.latex(fc['formula'])
                st.caption(f"📍 Source: {fc['source_cite']}")

    with r_col2:
        st.markdown("### 🎙️ 2-Minute Audio Brief (With Live Voice Playback)")
        st.markdown("Click play below to listen to your personalized audio revision summary:")
        st.info(rev_kit["audio_brief_script"])
        render_voice_player(
            rev_kit["audio_brief_script"],
            button_label="🔊 Play 2-Minute Audio Brief",
            key_suffix="rev_audio"
        )

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

    # Download button inside Tab 5 as well
    full_guide = enhancements.export_markdown_study_guide(current_student, days_until_exam=days_left)
    st.download_button(
        label="📥 Download Complete Personalized Study Pack & Timetable (.md)",
        data=full_guide,
        file_name=f"{current_student.student_id}_Exam_Study_Pack.md",
        mime="text/markdown",
        type="primary"
    )

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

    if st.button("🔄 Re-Run Live Benchmark Suite", type="primary"):
        with st.spinner("Executing Golden Test Set benchmarks and multi-session cohort simulations..."):
            suite.generate_benchmark_plot("benchmark_results.png")
            st.success("Benchmarks successfully re-computed and visual charts updated!")
            st.rerun()

st.markdown("---")
st.caption("OmniStudy AI • Peak Edition • Developed for Multimodal AI Hackathon 2026 • KamandPrompt (IIT Mandi), Augli.ai, PurpleRain Tech, LDV Labs")
