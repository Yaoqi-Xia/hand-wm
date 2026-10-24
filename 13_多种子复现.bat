@echo off
chcp 65001 >nul
cd /d %~dp0
call tools\find_python.bat
set PYTHONIOENCODING=utf-8
set LOG=runs\seeds_launcher_log.txt
echo Extra random seeds 1 and 2: same split, same 9 excluded clips, same 16 approved settings.
echo Finished trainings are skipped automatically. Please do not close this window.
echo [%date% %time%] launcher 13 started>> %LOG%
for %%S in (1 2) do (
  call :train main_s%%S --seed %%S || goto failed
  call :train mlp_s%%S --seed %%S --arch mlp || goto failed
  call :train action_s%%S --seed %%S --hand action || goto failed
)
echo === evaluation of the seed 1 and 2 models (same test clips as seed 0) ===
echo [%date% %time%] eval started>> %LOG%
"%PY%" scripts\eval_wm.py --predictors latest:main_s1,latest:main_s1:onestep,latest:mlp_s1,latest:action_s1,latest:main_s2,latest:main_s2:onestep,latest:mlp_s2,latest:action_s2 --out runs\eval_seeds12 || goto failed
echo [%date% %time%] eval finished>> %LOG%
echo === mean and spread over seeds 0, 1, 2 ===
"%PY%" scripts\aggregate_seeds.py runs\eval_20261019_133504\results.json runs\eval_seeds12\results.json || goto failed
echo [%date% %time%] ALL DONE>> %LOG%
echo.
echo ===== ALL DONE =====
goto end

:train
"%PY%" scripts\run_done.py %1 && exit /b 0
echo === training %* ===
echo [%date% %time%] training %1 started>> %LOG%
"%PY%" scripts\train_wm.py --tag %*
if errorlevel 1 (echo [%date% %time%] training %1 FAILED>> %LOG% & exit /b 1)
echo [%date% %time%] training %1 finished>> %LOG%
exit /b 0

:failed
echo [%date% %time%] STOPPED WITH AN ERROR - see this window>> %LOG%
echo.
echo ===== STOPPED WITH AN ERROR - the message above shows what happened. Double-click 13 again to continue. =====

:end
pause