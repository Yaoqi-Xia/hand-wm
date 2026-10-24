@echo off
chcp 65001 >nul
cd /d %~dp0
call tools\find_python.bat
echo === 1/4 main model: Transformer + 21 hand keypoints (about 35 min) ===
"%PY%" scripts\train_wm.py --tag main
if errorlevel 1 goto end
echo === 2/4 comparison: MLP (about 15 min) ===
"%PY%" scripts\train_wm.py --arch mlp --tag mlp
if errorlevel 1 goto end
echo === 3/4 comparison: hand as one 9-number token (about 30 min) ===
"%PY%" scripts\train_wm.py --hand action --tag action
if errorlevel 1 goto end
echo === 4/4 evaluation of everything ===
"%PY%" scripts\eval_wm.py --predictors static,const_velocity,latest:main,latest:main:onestep,latest:mlp,latest:action
:end
pause