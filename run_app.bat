@echo off
title OmniStudy AI Launcher
echo ===================================================
echo   Starting OmniStudy AI Study Companion...
echo   Track D: Personalized Tutoring & Adaptive Learning
echo ===================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
