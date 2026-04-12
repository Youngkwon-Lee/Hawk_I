@echo off
echo ============================================================
echo Hand Movement / Leg Agility Data Quality Analysis
echo ============================================================

cd C:\Users\YK\tulip\Hawkeye

echo.
echo [1/2] Running Statistical Analysis...
python scripts\analysis\visualize_hand_leg_quality.py

echo.
echo [2/2] Running Video Sample Visualization...
python scripts\analysis\visualize_video_samples.py

echo.
echo ============================================================
echo Analysis Complete!
echo ============================================================
echo Output: scripts\analysis\output\
pause
