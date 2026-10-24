@echo off
chcp 65001 >nul
cd /d %~dp0
call tools\find_python.bat
echo === 1/3 generate clips (resumable) ===
"%PY%" scripts\gen_dataset.py
if errorlevel 1 goto end
echo === 2/3 block-free hand inputs (resumable) ===
"%PY%" scripts\prep_hand_only.py
if errorlevel 1 goto end
echo === 3/3 physics report ===
"%PY%" scripts\physics_report.py
:end
pause
