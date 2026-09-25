@echo off
title VRIXA AI ASSISTANT - LIVE TUNNEL
cd /d "%~dp0"
echo ===================================================
echo   STARTING VRIXA AI ASSISTANT + CLOUDFLARE LIVE TUNNEL
echo   Creator: Harsh
echo ===================================================
echo Starting Local Backend Server...
start "Vrixa Server" python main.py
timeout /t 3 /nobreak >nul
echo Starting Cloudflare Live Public HTTPS Tunnel...
echo Copy the https://....trycloudflare.com link displayed below:
echo ===================================================
.\cloudflared.exe tunnel --protocol http2 --url http://127.0.0.1:8000
pause
