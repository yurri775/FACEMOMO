@echo off
chcp 65001 >nul
echo ====================================================================
echo      DEMONSTRATION RAPIDE - Systeme de Statistiques
echo ====================================================================
echo.
echo Execution de quick_demo.py...
echo.

python quick_demo.py

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
