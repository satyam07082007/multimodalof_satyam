@echo off
title Haven AI - FastAPI Backend
echo ===================================================
echo Starting Multimodal Emotional Support FastAPI Backend
echo Port: 8000
echo Docs: http://127.0.0.1:8000/docs
echo ===================================================
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
