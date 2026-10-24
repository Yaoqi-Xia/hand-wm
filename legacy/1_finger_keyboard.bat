@echo off
chcp 65001 >nul
cd /d %~dp0..
call tools\find_python.bat
"%PY%" scripts\teleop.py
pause
