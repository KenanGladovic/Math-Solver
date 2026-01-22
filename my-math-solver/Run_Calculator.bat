@echo off
cd /d "%~dp0"
:: --server.address=localhost forces the app to only exist inside your computer
py -m streamlit run App.py --browser.gatherUsageStats=false --server.headless=true --server.address=localhost
pause