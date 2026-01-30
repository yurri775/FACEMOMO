@echo off
chcp 65001 >nul
echo ====================================================================
echo      ANALYSE STATISTIQUE - Morphings Faciaux
echo ====================================================================
echo.
echo Ce script va analyser vos morphings avec le systeme de statistiques
echo base sur le papier SynMorph.
echo.
echo ====================================================================
echo.
echo Choisissez une option:
echo.
echo 1. Analyser les echantillons de demonstration
echo 2. Analyser vos morphings generes (morphing_results)
echo 3. Test rapide (20 images max)
echo.
set /p choice="> "

if "%choice%"=="1" (
    echo.
    echo Analyse des echantillons de demonstration...
    echo.
    python analyze_morphs.py --morph sample_data/after_morph --bona-fide sample_data/before_morph
) else if "%choice%"=="2" (
    echo.
    echo Analyse de morphing_results...
    echo.
    python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph
) else if "%choice%"=="3" (
    echo.
    echo Test rapide (20 images max)...
    echo.
    python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph --max 20
) else (
    echo.
    echo Choix invalide!
    pause
    exit /b
)

echo.
echo ====================================================================
echo Analyse terminee!
echo.
echo Les resultats sont dans: statistics_output\
echo ====================================================================
echo.
pause

echo Voulez-vous ouvrir le dossier des resultats? (O/N)
set /p open_choice="> "

if /i "%open_choice%"=="O" (
    explorer statistics_output
)
