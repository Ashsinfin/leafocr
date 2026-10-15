@echo off
REM ==========================================================
REM Palm Leaf OCR launcher (conda version)
REM Double-click this file (or a shortcut to it) to start the app.
REM ==========================================================

REM Always run from the folder containing this launcher.
cd /d "%~dp0"

REM Instead of "conda activate" (which can pick up the WRONG python.exe
REM when another Python 3.11 install is also on PATH, causing DLL
REM conflicts), call the palmleaf environment's own python.exe directly.
set PYTHONNOUSERSITE=1

echo Starting Palm Leaf OCR...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    echo Local environment not found; trying Python from PATH...
    python app.py
)

REM Keep the window open if it crashes, so you can read the error
pause
