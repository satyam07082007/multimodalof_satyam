@echo off
title HAVEN AI - Intelligent Multimodal Assistant (Frontend)
echo ===================================================
echo Starting HAVEN AI - One AI. Every Possibility.
echo Frontend Port: 8502
echo Live Web Link: http://localhost:8502
echo ===================================================
.\.venv\Scripts\streamlit.exe run frontend/streamlit_app.py --server.port 8502
pause
