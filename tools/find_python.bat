@echo off
rem Sets PY to the Python of the "handwm" environment.
rem Order: %HANDWM_PYTHON%, the path in local_python.txt (not tracked by git),
rem an activated conda env, common conda install locations, then plain "python".
set "PY="
if defined HANDWM_PYTHON set "PY=%HANDWM_PYTHON%"
if not defined PY if exist "%~dp0..\local_python.txt" set /p PY=<"%~dp0..\local_python.txt"
if not defined PY if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY for %%R in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
  if not defined PY if exist "%%~R\envs\handwm\python.exe" set "PY=%%~R\envs\handwm\python.exe"
)
if not defined PY set "PY=python"