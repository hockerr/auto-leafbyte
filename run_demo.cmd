@echo off
REM ===================================================================
REM  auto-leafbyte demo: batch-measure the 30 paper photos, then show
REM  the summary.  Open out\paper\review\index.html for the human review.
REM
REM  Needs Python 3.10+ with numpy and Pillow on PATH (or set PY below).
REM  First run: fetch the upstream test images once with
REM      python tools\fetch_paper_data.py
REM ===================================================================
setlocal
set "ROOT=%~dp0"
if not defined PY set "PY=python"
set "PYTHONPATH=%ROOT%src"

if not exist "%ROOT%data\paper\images\CO1.png" (
    echo [0/3] fetching upstream test images ^(once, ~57 MB^) ...
    "%PY%" "%ROOT%tools\fetch_paper_data.py"
    if errorlevel 1 goto :error
)

echo [1/3] measuring data\paper\images ...
"%PY%" -m leafbyte run "%ROOT%data\paper\images" --out "%ROOT%out\paper" --scale-cm 17
if errorlevel 1 goto :error

echo.
echo [2/3] summary
"%PY%" -m leafbyte report --out "%ROOT%out\paper" -v

echo.
echo ===============================================================
echo  Review page : %ROOT%out\paper\review\index.html
echo  Results     : %ROOT%out\paper\results.csv
echo  Next: open the review page, export decisions.json, then run
echo        "%PY%" -m leafbyte apply --out "%ROOT%out\paper" --decisions "...\decisions.json" --reviewer yourname
echo ===============================================================
pause
exit /b 0

:error
echo.
echo FAILED - see the messages above.
pause
exit /b 1
