@echo off
title Haven AI - Streamlit Interface
echo ===================================================
echo Starting Haven Emotional Support Streamlit Frontend
echo Port: 8501
echo Link: http://localhost:8501
echo ===================================================
.\.venv\Scripts\streamlit.exe run frontend/streamlit_app.py --server.port 8501
pause
