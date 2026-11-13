@echo off
REM ============================================================
REM Sistema de Supervision Inteligente - UN SOLO ARRANQUE
REM  1. La primera vez instala el entorno (.venv) y cloudflared.
REM  2. Cierra restos de un inicio anterior (puertos 8501/8502).
REM  3. Abre la pagina, la camara de ESTA laptop y un link publico
REM     (https://...trycloudflare.com) para verla desde cualquier red.
REM Cada inicio da un link y una clave NUEVOS. Al cerrar esta
REM ventana el link deja de funcionar (error 1033 de Cloudflare).
REM ============================================================
cd /d "%~dp0"
title Sistema de Supervision Inteligente

if not exist ".venv\Scripts\python.exe" (
    echo Preparando el entorno por primera vez ^(puede tardar varios minutos^)...
    py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo No se encontro Python. Instalalo desde https://www.python.org y vuelve a intentar.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requisitos.txt || (
        echo Fallo la instalacion de librerias.
        pause
        exit /b 1
    )
)

if not exist "herramientas\cloudflared.exe" (
    echo Descargando cloudflared desde la pagina oficial de Cloudflare...
    if not exist herramientas mkdir herramientas
    powershell -NoProfile -Command "Invoke-WebRequest -Uri https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile herramientas\cloudflared.exe"
)

echo Cerrando restos de un inicio anterior...
taskkill /f /im cloudflared.exe >nul 2>&1
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8501,8502 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
if exist "resultados\tunel.json" del "resultados\tunel.json"

".venv\Scripts\python.exe" integracion\publicar.py
pause
