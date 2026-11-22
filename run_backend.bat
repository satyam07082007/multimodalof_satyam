@echo off
title HAVEN AI - Backend API Services
echo ===================================================
echo Starting HAVEN AI - One AI. Every Possibility.
echo Backend API Port: 8000
echo Swagger Docs: http://127.0.0.1:8000/docs
echo ===================================================
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
