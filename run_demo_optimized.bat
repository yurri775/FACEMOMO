@echo off
chcp 65001 >nul
echo ====================================================================
echo      DEMONSTRATION RAPIDE - Systeme Optimise
echo ====================================================================
echo.
echo Execution de quick_demo_optimized.py...
echo Version professionnelle avec algorithmes avances
echo.

python quick_demo_optimized.py

echo.
echo ====================================================================
echo Demonstration terminee!
echo.
echo Les resultats sont dans: demo_statistics_output\
echo ====================================================================
echo.
pause

echo Voulez-vous ouvrir le dossier des resultats? (O/N)
set /p choice="> "

if /i "%choice%"=="O" (
    explorer demo_statistics_output
)
