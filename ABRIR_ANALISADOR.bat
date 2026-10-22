@echo off
chcp 65001 >nul
title Analisador de Acoes
cd /d "%~dp0"
echo Iniciando o Analisador de Acoes...
echo O navegador vai abrir em alguns segundos. Para encerrar, feche esta janela.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8501"
uv run streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause
