@echo off
title Asistente de IA de Unity
color 0A

cd /d C:\proyectos\UnityAssistant

tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul || start "" ollama serve

echo Iniciando Unity AI...

python unity_rag.py

pause