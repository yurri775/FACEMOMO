@echo off
chcp 65001 >nul
cls
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                                                                      ║
echo ║         SYSTEME OPTIMISE D'ANALYSE DE MORPHINGS FACIAUX             ║
echo ║                     Version Professionnelle                          ║
echo ║                                                                      ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo.
echo    Que voulez-vous faire?
echo.
echo    [1] DEMONSTRATION RAPIDE (30 secondes)
echo        - Genere des donnees de test realistes
echo        - Cree 4 graphiques professionnels
echo        - Produit un rapport complet sans emojis
echo.
echo    [2] ANALYSER MES MORPHINGS
echo        - Analyse complete de vos morphings
echo        - FIQA + MAP sur 4 systemes FRS
echo        - Statistiques robustes avec Bootstrap
echo.
echo    [3] TESTER LE MODULE (verification technique)
echo        - Test des algorithmes avances
echo        - Verification des methodes optimisees
echo.
echo    [4] AIDE - Ou sont mes fichiers?
echo.
echo    [0] QUITTER
echo.
echo ══════════════════════════════════════════════════════════════════════
echo.
set /p choice="Votre choix (1-4 ou 0): "

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto analyse
if "%choice%"=="3" goto test
if "%choice%"=="4" goto aide
if "%choice%"=="0" goto fin
goto invalide

:demo
cls
echo ══════════════════════════════════════════════════════════════════════
echo                       DEMONSTRATION RAPIDE
echo ══════════════════════════════════════════════════════════════════════
echo.
python quick_demo_optimized.py
echo.
echo ══════════════════════════════════════════════════════════════════════
echo.
echo RESULTATS dans: demo_statistics_output\
echo.
echo Fichiers generes:
echo   - demo_report_optimized.txt (rapport complet sans emojis)
echo   - 4 graphiques PNG professionnels
echo.
pause
echo.
echo Ouvrir le dossier des resultats? (O/N)
set /p open_demo="> "
if /i "%open_demo%"=="O" explorer demo_statistics_output
goto menu

:analyse
cls
echo ══════════════════════════════════════════════════════════════════════
echo                    ANALYSE DE VOS MORPHINGS
echo ══════════════════════════════════════════════════════════════════════
echo.
echo Options disponibles:
echo.
echo [1] Analyser les echantillons de demonstration
echo [2] Analyser morphing_results
echo [3] Test rapide (20 images max)
echo [0] Retour
echo.
set /p ana_choice="> "

if "%ana_choice%"=="1" (
    python analyze_morphs_optimized.py --morph sample_data/after_morph --bona-fide sample_data/before_morph
) else if "%ana_choice%"=="2" (
    python analyze_morphs_optimized.py --morph morphing_results --bona-fide sample_data/before_morph
) else if "%ana_choice%"=="3" (
    python analyze_morphs_optimized.py --morph morphing_results --bona-fide sample_data/before_morph --max 20
) else if "%ana_choice%"=="0" (
    goto menu
) else (
    echo Choix invalide!
    pause
    goto analyse
)

echo.
echo ══════════════════════════════════════════════════════════════════════
echo.
echo RESULTATS dans: statistics_output\
echo.
pause
echo.
echo Ouvrir le dossier des resultats? (O/N)
set /p open_stats="> "
if /i "%open_stats%"=="O" explorer statistics_output
goto menu

:test
cls
echo ══════════════════════════════════════════════════════════════════════
echo                    TEST DU MODULE OPTIMISE
echo ══════════════════════════════════════════════════════════════════════
echo.
echo Execution de statistics_module_optimized.py...
echo (Test des algorithmes: Gradient Descent, Bootstrap, Monte Carlo, PCA)
echo.
python statistics_module_optimized.py
echo.
pause
goto menu

:aide
cls
echo ══════════════════════════════════════════════════════════════════════
echo                            GUIDE D'AIDE
echo ══════════════════════════════════════════════════════════════════════
echo.
echo STRUCTURE DES FICHIERS:
echo.
echo Dossier principal: c:\Users\marwa\OneDrive\Desktop\moprh\
echo.
echo FICHIERS OPTIMISES (RECOMMANDES - Sans emojis):
echo   - quick_demo_optimized.py           Demo rapide
echo   - analyze_morphs_optimized.py       Analyse complete
echo   - statistics_module_optimized.py    Module principal (750 lignes)
echo   - run_demo_optimized.bat            Batch demo
echo   - run_analysis_optimized.bat        Batch analyse
echo   - LANCEZ_MOI.bat                    Ce fichier (menu principal)
echo.
echo FICHIERS ORIGINAUX (Avec emojis - obsoletes):
echo   - quick_demo.py
echo   - analyze_morphs.py
echo   - statistics_module.py
echo.
echo RESULTATS:
echo   - demo_statistics_output\           Resultats de la demo
echo   - statistics_output\                Resultats de l'analyse
echo.
echo IMPORTANT:
echo   Vous devez TOUJOURS etre dans le dossier:
echo   c:\Users\marwa\OneDrive\Desktop\moprh\
echo.
echo   Si vous etes ailleurs, tapez:
echo   cd c:\Users\marwa\OneDrive\Desktop\moprh
echo.
pause
goto menu

:invalide
echo.
echo Choix invalide! Veuillez choisir 1, 2, 3, 4 ou 0.
pause
goto menu

:menu
cls
goto :eof

:fin
echo.
echo Au revoir!
timeout /t 2 >nul
exit
