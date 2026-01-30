# -*- coding: utf-8 -*-
"""
Demonstration Rapide du Systeme de Statistiques Optimise
Version professionnelle sans emojis avec resultats realistes
"""

import sys
import os

# Configuration de l'encodage UTF-8 pour Windows
if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from pathlib import Path
from statistics_module_optimized import OptimizedFIQAAnalyzer, OptimizedMAPAnalyzer, OptimizedStatisticsVisualizer


def create_realistic_demo_morphs():
    """
    Cree des morphings de demonstration avec variations realistes
    Simule les caracteristiques des vrais morphings faciaux
    """
    print("\n" + "="*80)
    print("CREATION DES DONNEES DE DEMONSTRATION")
    print("="*80 + "\n")

    output_dir = Path("demo_statistics_output")
    output_dir.mkdir(exist_ok=True)

    morphs_dir = output_dir / "demo_morphs"
    bona_fide_dir = output_dir / "demo_bona_fide"
    morphs_dir.mkdir(exist_ok=True)
    bona_fide_dir.mkdir(exist_ok=True)

    np.random.seed(42)  # Pour reproductibilite

    print("Generation des images morphees (qualite reduite)...")
    # Les morphings ont generalement une qualite reduite due au processus de melange
    for i in range(10):
        # Creation d'image avec variabilite realiste
        base_brightness = np.random.uniform(100, 180)
        noise_level = np.random.uniform(15, 35)  # Plus de bruit pour morphs

        img = np.random.normal(base_brightness, noise_level, (224, 224, 3))

        # Ajouter des artefacts de morphing realistes
        # 1. Reduction de nettete (blurring)
        blur_kernel = np.random.randint(3, 7)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)

        # 2. Artefacts de compression
        compression_noise = np.random.normal(0, 5, img.shape)
        img = img + compression_noise

        # 3. Variations additionnelles (bruit supplementaire pour certaines images)
        if i % 3 == 0:
            extra_noise = np.random.normal(0, 10, img.shape)
            img = img + extra_noise

        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(morphs_dir / f"morph_{i:03d}.png"), img)

    print(f"  - {len(list(morphs_dir.glob('*.png')))} images morphees creees")

    print("\nGeneration des images bona fide (qualite elevee)...")
    # Les images authentiques ont une meilleure qualite
    for i in range(10):
        base_brightness = np.random.uniform(110, 200)
        noise_level = np.random.uniform(5, 15)  # Moins de bruit

        img = np.random.normal(base_brightness, noise_level, (224, 224, 3))

        # Moins de flou pour les images authentiques
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Details plus nets
        img = cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), 0, 5)

        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(bona_fide_dir / f"bonafide_{i:03d}.png"), img)

    print(f"  - {len(list(bona_fide_dir.glob('*.png')))} images bona fide creees\n")

    return morphs_dir, bona_fide_dir


def run_comprehensive_demo():
    """Execute une demonstration complete du systeme de statistiques"""

    print("\n" + "="*80)
    print("SYSTEME D'ANALYSE STATISTIQUE - FACE MORPHING")
    print("Base sur SynMorph (arXiv:2409.05595v1)")
    print("Version Optimisee avec Algorithmes Avances")
    print("="*80 + "\n")

    # Creation des donnees de test
    morphs_dir, bona_fide_dir = create_realistic_demo_morphs()
    output_dir = Path("demo_statistics_output")

    # ===================================================================
    # PARTIE 1: FIQA - Face Image Quality Assessment
    # ===================================================================
    print("\n" + "="*80)
    print("PHASE 1: FACE IMAGE QUALITY ASSESSMENT (FIQA)")
    print("="*80 + "\n")

    print("Initialisation de l'analyseur FIQA...")
    fiqa_analyzer = OptimizedFIQAAnalyzer()

    print("\nAnalyse des images morphees...")
    print("  - Extraction de features avancees (10+ metriques)")
    print("  - Calcul de statistiques robustes")
    print("  - Bootstrap pour intervalles de confiance")
    morph_stats = fiqa_analyzer.analyze_batch(list(morphs_dir.glob("*.png")))

    print(f"\n  Resultats morphs:")
    print(f"    Moyenne:    {morph_stats['mean']:.4f}")
    print(f"    Ecart-type: {morph_stats['std']:.4f}")
    print(f"    Mediane:    {morph_stats['median']:.4f}")
    print(f"    IC 95%%:     [{morph_stats['ci_lower']:.4f}, {morph_stats['ci_upper']:.4f}]")

    print("\nAnalyse des images bona fide...")
    bf_stats = fiqa_analyzer.analyze_batch(list(bona_fide_dir.glob("*.png")))

    print(f"\n  Resultats bona fide:")
    print(f"    Moyenne:    {bf_stats['mean']:.4f}")
    print(f"    Ecart-type: {bf_stats['std']:.4f}")
    print(f"    Mediane:    {bf_stats['median']:.4f}")
    print(f"    IC 95%%:     [{bf_stats['ci_lower']:.4f}, {bf_stats['ci_upper']:.4f}]")

    # Calcul KL-Divergence
    print("\nCalcul de la divergence KL entre distributions...")
    kl_div = fiqa_analyzer.compute_kl_divergence(
        morph_stats['scores'],
        bf_stats['scores']
    )
    print(f"  KL-Divergence: {kl_div:.6f}")

    # ===================================================================
    # PARTIE 2: MAP - Morphing Attack Potential
    # ===================================================================
    print("\n" + "="*80)
    print("PHASE 2: MORPHING ATTACK POTENTIAL (MAP)")
    print("ISO/IEC 20059 Standard")
    print("="*80 + "\n")

    print("Initialisation de l'analyseur MAP...")
    map_analyzer = OptimizedMAPAnalyzer()

    # Chargement des images pour MAP
    morph_images = [cv2.imread(str(p)) for p in morphs_dir.glob("*.png")]
    bf_images = [cv2.imread(str(p)) for p in bona_fide_dir.glob("*.png")]

    # Test sur 4 systemes FRS
    frs_systems = ['arcface', 'dlib', 'facenet', 'vggface']
    map_results = {}

    print("Test sur 4 systemes de reconnaissance faciale:")
    for i, frs in enumerate(frs_systems, 1):
        print(f"\n  [{i}/4] {frs.upper()}...")

        # Simulation de mated samples (2 sujets A et B)
        mid = len(bf_images) // 2
        mated_a = bf_images[:mid]
        mated_b = bf_images[mid:]

        print(f"    - Calcul de similarites avec Monte Carlo...")
        result = map_analyzer.compute_map(
            morph_images[:5],  # Limiter pour demo rapide
            mated_a[:5],
            mated_b[:5],
            frs_system=frs
        )

        map_results[frs] = result
        print(f"    - MAP Score: {result['map_score']:.4f}")
        print(f"    - Matches A: {result['matches_a']}")
        print(f"    - Matches B: {result['matches_b']}")

    # Calcul MAP moyen
    avg_map = np.mean([r['map_score'] for r in map_results.values()])
    print(f"\n  MAP Moyen sur tous les systemes: {avg_map:.4f}")

    # ===================================================================
    # PARTIE 3: VISUALISATIONS
    # ===================================================================
    print("\n" + "="*80)
    print("PHASE 3: GENERATION DES VISUALISATIONS")
    print("="*80 + "\n")

    print("Initialisation du module de visualisation...")
    visualizer = OptimizedStatisticsVisualizer()

    print("\nGeneration des graphiques:")

    # KDE Plot
    print("  [1/4] Kernel Density Estimation (KDE) Plot...")
    stats_dict = {
        'morph': morph_stats,
        'bona_fide': bf_stats,
        'kl_divergence': kl_div
    }
    kde_path = output_dir / "fiqa_kde_optimized.png"
    visualizer.plot_kde_with_ci(stats_dict, str(kde_path))
    print(f"        Sauvegarde: {kde_path.name}")

    # MAP Comparison
    print("  [2/4] Morphing Attack Potential Comparison...")
    map_path = output_dir / "map_comparison_optimized.png"
    visualizer.plot_map_comparison(map_results, str(map_path))
    print(f"        Sauvegarde: {map_path.name}")

    # DET Curve (simule)
    print("  [3/4] Detection Error Tradeoff (DET) Curve...")
    # Generation de scores simules pour DET
    genuine_scores = np.random.beta(8, 2, 100)  # Distribution pour genuines
    impostor_scores = np.random.beta(2, 5, 100)  # Distribution pour impostors

    det_path = output_dir / "det_curve_optimized.png"
    visualizer.plot_det_curve(genuine_scores, impostor_scores, str(det_path))
    print(f"        Sauvegarde: {det_path.name}")

    # Comparaison multi-methodes
    print("  [4/4] Comparaison des methodes FIQA...")
    multi_stats = {
        'Simple': stats_dict,
        'FaceQnet': stats_dict,  # Simule pour demo
        'SER-FIQ': stats_dict    # Simule pour demo
    }
    comp_path = output_dir / "fiqa_methods_comparison_optimized.png"
    visualizer.plot_fiqa_methods_comparison(multi_stats, str(comp_path))
    print(f"        Sauvegarde: {comp_path.name}")

    # ===================================================================
    # PARTIE 4: RAPPORT TEXTE
    # ===================================================================
    print("\n" + "="*80)
    print("PHASE 4: GENERATION DU RAPPORT")
    print("="*80 + "\n")

    report_path = output_dir / "demo_report_optimized.txt"
    print(f"Ecriture du rapport complet: {report_path.name}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("                 RAPPORT D'ANALYSE STATISTIQUE\n")
        f.write("              Face Morphing - Base sur SynMorph\n")
        f.write("              Version Optimisee avec Algorithmes Avances\n")
        f.write("="*80 + "\n\n")

        f.write("FACE IMAGE QUALITY ASSESSMENT (FIQA)\n")
        f.write("-"*80 + "\n\n")

        f.write("Images Morphees:\n")
        f.write(f"  Moyenne:     {morph_stats['mean']:.4f}\n")
        f.write(f"  Ecart-type:  {morph_stats['std']:.4f}\n")
        f.write(f"  Min:         {morph_stats['min']:.4f}\n")
        f.write(f"  Max:         {morph_stats['max']:.4f}\n")
        f.write(f"  Mediane:     {morph_stats['median']:.4f}\n")
        f.write(f"  Q25:         {morph_stats['q25']:.4f}\n")
        f.write(f"  Q75:         {morph_stats['q75']:.4f}\n")
        f.write(f"  Skewness:    {morph_stats['skewness']:.4f}\n")
        f.write(f"  Kurtosis:    {morph_stats['kurtosis']:.4f}\n")
        f.write(f"  IC 95%%:      [{morph_stats['ci_lower']:.4f}, {morph_stats['ci_upper']:.4f}]\n")
        f.write(f"  Count:       {len(morph_stats['scores'])}\n\n")

        f.write("Images Bona Fide:\n")
        f.write(f"  Moyenne:     {bf_stats['mean']:.4f}\n")
        f.write(f"  Ecart-type:  {bf_stats['std']:.4f}\n")
        f.write(f"  Min:         {bf_stats['min']:.4f}\n")
        f.write(f"  Max:         {bf_stats['max']:.4f}\n")
        f.write(f"  Mediane:     {bf_stats['median']:.4f}\n")
        f.write(f"  Q25:         {bf_stats['q25']:.4f}\n")
        f.write(f"  Q75:         {bf_stats['q75']:.4f}\n")
        f.write(f"  Skewness:    {bf_stats['skewness']:.4f}\n")
        f.write(f"  Kurtosis:    {bf_stats['kurtosis']:.4f}\n")
        f.write(f"  IC 95%%:      [{bf_stats['ci_lower']:.4f}, {bf_stats['ci_upper']:.4f}]\n")
        f.write(f"  Count:       {len(bf_stats['scores'])}\n\n")

        f.write(f"Divergence de Kullback-Leibler: {kl_div:.6f}\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("MORPHING ATTACK POTENTIAL (MAP)\n")
        f.write("-"*80 + "\n\n")

        for frs, result in map_results.items():
            f.write(f"{frs.upper()}:\n")
            f.write(f"  MAP Score:    {result['map_score']:.4f}\n")
            f.write(f"  Matches A:    {result['matches_a']}\n")
            f.write(f"  Matches B:    {result['matches_b']}\n")
            f.write(f"  Total Comps:  {result['total_comparisons']}\n\n")

        f.write(f"MAP Moyen: {avg_map:.4f}\n\n")

        f.write("="*80 + "\n")
        f.write("ALGORITHMES AVANCES UTILISES\n")
        f.write("-"*80 + "\n\n")
        f.write("  - Gradient Descent avec regularisation L2\n")
        f.write("  - Optimisation Convexe (analytique)\n")
        f.write("  - Bootstrap Method (1000 iterations)\n")
        f.write("  - Monte Carlo Estimation\n")
        f.write("  - Principal Component Analysis (PCA)\n")
        f.write("  - Cubic Spline Approximation\n")
        f.write("  - Multi-scale Feature Extraction\n")
        f.write("  - Kernel Density Estimation\n\n")

        f.write("="*80 + "\n")
        f.write("Fin du rapport\n")
        f.write("="*80 + "\n")

    print(f"  Rapport sauvegarde avec succes\n")

    # ===================================================================
    # RESUME FINAL
    # ===================================================================
    print("\n" + "="*80)
    print("DEMONSTRATION TERMINEE AVEC SUCCES")
    print("="*80 + "\n")

    print("Fichiers generes dans: demo_statistics_output/")
    print("\n  Visualisations:")
    print("    - fiqa_kde_optimized.png")
    print("    - map_comparison_optimized.png")
    print("    - det_curve_optimized.png")
    print("    - fiqa_methods_comparison_optimized.png")
    print("\n  Rapport:")
    print("    - demo_report_optimized.txt")
    print("\n  Donnees de test:")
    print(f"    - demo_morphs/        ({len(list(morphs_dir.glob('*.png')))} images)")
    print(f"    - demo_bona_fide/     ({len(list(bona_fide_dir.glob('*.png')))} images)")

    print("\n" + "="*80)
    print("Pour analyser vos propres morphings:")
    print("  python analyze_morphs_optimized.py --morph <dossier> --bona-fide <dossier>")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        run_comprehensive_demo()
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
