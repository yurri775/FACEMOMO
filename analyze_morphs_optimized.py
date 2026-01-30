# -*- coding: utf-8 -*-
"""
Script d'Analyse de Morphings Faciaux - Version Optimisee
==========================================================

Analyse complete de morphings faciaux avec:
1. Detection automatique des dossiers
2. Analyse FIQA (Face Image Quality Assessment) avancee
3. Calcul MAP (Morphing Attack Potential) sur 4 systemes FRS
4. Visualisations professionnelles (KDE, DET, comparaisons)
5. Rapport detaille avec statistiques robustes

Version optimisee sans emojis avec algorithmes avances
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
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics_module_optimized import OptimizedFIQAAnalyzer, OptimizedMAPAnalyzer, OptimizedStatisticsVisualizer


def validate_image_folder(folder_path, min_images=5):
    """
    Valide qu'un dossier contient suffisamment d'images valides

    Args:
        folder_path: Chemin du dossier a valider
        min_images: Nombre minimum d'images requises

    Returns:
        tuple: (is_valid, image_paths, error_message)
    """
    folder = Path(folder_path)

    if not folder.exists():
        return False, [], f"Le dossier n'existe pas: {folder_path}"

    if not folder.is_dir():
        return False, [], f"Le chemin n'est pas un dossier: {folder_path}"

    # Extensions d'images supportees
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Collecter toutes les images valides
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(folder.glob(f"*{ext}"))
        image_paths.extend(folder.glob(f"*{ext.upper()}"))

    if len(image_paths) < min_images:
        return False, [], f"Pas assez d'images (trouve: {len(image_paths)}, requis: {min_images})"

    return True, sorted(image_paths), None


def load_images_batch(image_paths, max_images=None, show_progress=True):
    """
    Charge un batch d'images avec verification et barre de progression

    Args:
        image_paths: Liste de chemins d'images
        max_images: Limite optionnelle du nombre d'images
        show_progress: Afficher barre de progression

    Returns:
        list: Images chargees avec succes
    """
    if max_images:
        image_paths = image_paths[:max_images]

    images = []
    failed = 0

    iterator = tqdm(image_paths, desc="Chargement", disable=not show_progress)

    for img_path in iterator:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
            else:
                failed += 1
        except Exception:
            failed += 1

    if failed > 0:
        print(f"  Attention: {failed} image(s) n'ont pas pu etre chargees")

    return images


def analyze_face_morphing_dataset(morph_dir, bona_fide_dir, output_dir="statistics_output",
                                  max_images=None, frs_systems=None):
    """
    Analyse complete d'un dataset de morphings faciaux

    Args:
        morph_dir: Dossier contenant les images morphees
        bona_fide_dir: Dossier contenant les images authentiques
        output_dir: Dossier de sortie pour resultats
        max_images: Limite optionnelle du nombre d'images
        frs_systems: Liste des systemes FRS a tester
    """

    print("\n" + "="*80)
    print("ANALYSE STATISTIQUE DE MORPHINGS FACIAUX")
    print("Base sur SynMorph (arXiv:2409.05595v1)")
    print("Version Optimisee - Algorithmes Avances")
    print("="*80 + "\n")

    # ===================================================================
    # VALIDATION DES ENTREES
    # ===================================================================
    print("VALIDATION DES DONNEES")
    print("-"*80 + "\n")

    print(f"Dossier morphs:     {morph_dir}")
    valid, morph_paths, error = validate_image_folder(morph_dir)
    if not valid:
        print(f"  ERREUR: {error}")
        return False
    print(f"  OK - {len(morph_paths)} images trouvees")

    print(f"\nDossier bona fide:  {bona_fide_dir}")
    valid, bf_paths, error = validate_image_folder(bona_fide_dir)
    if not valid:
        print(f"  ERREUR: {error}")
        return False
    print(f"  OK - {len(bf_paths)} images trouvees")

    # Creation du dossier de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"\nDossier de sortie:  {output_dir}")
    print(f"  Cree avec succes\n")

    # ===================================================================
    # CHARGEMENT DES IMAGES
    # ===================================================================
    print("="*80)
    print("CHARGEMENT DES IMAGES")
    print("="*80 + "\n")

    print("Chargement des images morphees...")
    morph_images = load_images_batch(morph_paths, max_images)
    print(f"  {len(morph_images)} images chargees avec succes\n")

    print("Chargement des images bona fide...")
    bf_images = load_images_batch(bf_paths, max_images)
    print(f"  {len(bf_images)} images chargees avec succes\n")

    if len(morph_images) < 5 or len(bf_images) < 5:
        print("ERREUR: Pas assez d'images chargees pour analyse statistique")
        print("Minimum requis: 5 images de chaque type")
        return False

    # ===================================================================
    # ANALYSE FIQA
    # ===================================================================
    print("="*80)
    print("PHASE 1: FACE IMAGE QUALITY ASSESSMENT (FIQA)")
    print("="*80 + "\n")

    print("Initialisation de l'analyseur FIQA avec algorithmes avances...")
    print("  - Extraction de features multi-echelles")
    print("  - Bootstrap pour intervalles de confiance (1000 iterations)")
    print("  - Analyse en composantes principales (PCA)")
    print("  - Optimisation convexe pour ponderation\n")

    fiqa_analyzer = OptimizedFIQAAnalyzer()

    print(f"Analyse des {len(morph_images)} images morphees...")
    morph_stats = fiqa_analyzer.analyze_batch(
        morph_paths[:len(morph_images)],
        show_progress=True
    )

    print(f"\nResultats - Images Morphees:")
    print(f"  Moyenne:     {morph_stats['mean']:.4f} (IC 95%: [{morph_stats['ci_lower']:.4f}, {morph_stats['ci_upper']:.4f}])")
    print(f"  Ecart-type:  {morph_stats['std']:.4f}")
    print(f"  Mediane:     {morph_stats['median']:.4f}")
    print(f"  Min - Max:   {morph_stats['min']:.4f} - {morph_stats['max']:.4f}")
    print(f"  Skewness:    {morph_stats['skewness']:.4f}")
    print(f"  Kurtosis:    {morph_stats['kurtosis']:.4f}")

    print(f"\nAnalyse des {len(bf_images)} images bona fide...")
    bf_stats = fiqa_analyzer.analyze_batch(
        bf_paths[:len(bf_images)],
        show_progress=True
    )

    print(f"\nResultats - Images Bona Fide:")
    print(f"  Moyenne:     {bf_stats['mean']:.4f} (IC 95%: [{bf_stats['ci_lower']:.4f}, {bf_stats['ci_upper']:.4f}])")
    print(f"  Ecart-type:  {bf_stats['std']:.4f}")
    print(f"  Mediane:     {bf_stats['median']:.4f}")
    print(f"  Min - Max:   {bf_stats['min']:.4f} - {bf_stats['max']:.4f}")
    print(f"  Skewness:    {bf_stats['skewness']:.4f}")
    print(f"  Kurtosis:    {bf_stats['kurtosis']:.4f}")

    # Calcul KL-Divergence
    print("\nCalcul de la divergence KL entre les distributions...")
    kl_div = fiqa_analyzer.compute_kl_divergence(
        morph_stats['scores'],
        bf_stats['scores']
    )
    print(f"  KL-Divergence: {kl_div:.6f}")
    print(f"  (Mesure de separation des distributions)")

    # Test statistique
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_ind(morph_stats['scores'], bf_stats['scores'])
    print(f"\nTest t de Student:")
    print(f"  t-statistique: {t_stat:.4f}")
    print(f"  p-value:       {p_value:.6f}")
    if p_value < 0.05:
        print(f"  Les distributions sont significativement differentes (p < 0.05)")
    else:
        print(f"  Pas de difference significative (p >= 0.05)")

    # ===================================================================
    # ANALYSE MAP
    # ===================================================================
    print("\n" + "="*80)
    print("PHASE 2: MORPHING ATTACK POTENTIAL (MAP)")
    print("Standard ISO/IEC 20059")
    print("="*80 + "\n")

    print("Initialisation de l'analyseur MAP...")
    print("  - Extraction d'embeddings multi-systemes")
    print("  - Calcul de similarites (cosine, euclidean, Pearson)")
    print("  - Estimation Monte Carlo pour robustesse\n")

    map_analyzer = OptimizedMAPAnalyzer()

    # Preparation des mated samples (simuler 2 sujets)
    mid = len(bf_images) // 2
    mated_samples_a = bf_images[:mid]
    mated_samples_b = bf_images[mid:]

    print(f"Configuration MAP:")
    print(f"  Morphs:          {len(morph_images)} images")
    print(f"  Mated samples A: {len(mated_samples_a)} images")
    print(f"  Mated samples B: {len(mated_samples_b)} images\n")

    # Systemes FRS a tester
    if frs_systems is None:
        frs_systems = ['arcface', 'dlib', 'facenet', 'vggface']

    map_results = {}

    print(f"Test sur {len(frs_systems)} systemes de reconnaissance faciale:\n")

    for i, frs in enumerate(frs_systems, 1):
        print(f"  [{i}/{len(frs_systems)}] {frs.upper()}")

        # Limiter a 30 morphs pour performance (ajustable)
        max_map_morphs = min(30, len(morph_images))
        max_mated = min(15, len(mated_samples_a), len(mated_samples_b))

        print(f"       Analyse de {max_map_morphs} morphs vs {max_mated} mated samples...")

        result = map_analyzer.compute_map(
            morph_images[:max_map_morphs],
            mated_samples_a[:max_mated],
            mated_samples_b[:max_mated],
            frs_system=frs
        )

        map_results[frs] = result

        print(f"       MAP Score:     {result['map_score']:.4f}")
        print(f"       Matches A:     {result['matches_a']}/{result['total_comparisons']}")
        print(f"       Matches B:     {result['matches_b']}/{result['total_comparisons']}")
        print(f"       Attack Rate:   {result['map_score']*100:.2f}%\n")

    # Calcul MAP moyen
    avg_map = np.mean([r['map_score'] for r in map_results.values()])
    std_map = np.std([r['map_score'] for r in map_results.values()])

    print(f"  Resultats Globaux:")
    print(f"    MAP Moyen:      {avg_map:.4f} +/- {std_map:.4f}")
    print(f"    MAP Min:        {min(r['map_score'] for r in map_results.values()):.4f}")
    print(f"    MAP Max:        {max(r['map_score'] for r in map_results.values()):.4f}")

    # ===================================================================
    # VISUALISATIONS
    # ===================================================================
    print("\n" + "="*80)
    print("PHASE 3: GENERATION DES VISUALISATIONS")
    print("="*80 + "\n")

    print("Initialisation du module de visualisation...\n")
    visualizer = OptimizedStatisticsVisualizer()

    stats_dict = {
        'morph': morph_stats,
        'bona_fide': bf_stats,
        'kl_divergence': kl_div
    }

    print("Generation des graphiques:")

    # 1. KDE Plot
    print("  [1/4] Kernel Density Estimation avec intervalles de confiance...")
    kde_path = output_path / "fiqa_kde_distribution.png"
    visualizer.plot_kde_with_ci(stats_dict, str(kde_path))
    print(f"        Sauvegarde: {kde_path.name}")

    # 2. MAP Comparison
    print("  [2/4] Comparaison MAP entre systemes FRS...")
    map_path = output_path / "map_comparison.png"
    visualizer.plot_map_comparison(map_results, str(map_path))
    print(f"        Sauvegarde: {map_path.name}")

    # 3. DET Curve
    print("  [3/4] Detection Error Tradeoff (DET) Curve...")
    det_path = output_path / "det_curve.png"

    # Utiliser scores FIQA pour DET
    genuine_scores = np.array(bf_stats['scores'])
    impostor_scores = np.array(morph_stats['scores'])

    visualizer.plot_det_curve(genuine_scores, impostor_scores, str(det_path))
    print(f"        Sauvegarde: {det_path.name}")

    # 4. Box Plot comparatif
    print("  [4/4] Box Plot comparatif des distributions...")
    box_path = output_path / "fiqa_boxplot_comparison.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [morph_stats['scores'], bf_stats['scores']]
    box = ax.boxplot(data_to_plot, labels=['Morphed', 'Bona Fide'], patch_artist=True)

    # Couleurs
    colors = ['#ff6b6b', '#4ecdc4']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('FIQA Score')
    ax.set_title('Distribution Comparison - FIQA Scores')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(box_path), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"        Sauvegarde: {box_path.name}")

    # ===================================================================
    # RAPPORT TEXTE
    # ===================================================================
    print("\n" + "="*80)
    print("PHASE 4: GENERATION DU RAPPORT TEXTE")
    print("="*80 + "\n")

    report_path = output_path / "analysis_report.txt"
    print(f"Ecriture du rapport complet: {report_path.name}\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("                 RAPPORT D'ANALYSE STATISTIQUE\n")
        f.write("                  Morphing Attack Detection\n")
        f.write("              Base sur SynMorph (arXiv:2409.05595v1)\n")
        f.write("="*80 + "\n\n")

        # Dataset info
        f.write("INFORMATIONS SUR LE DATASET\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Dossier morphs:      {morph_dir}\n")
        f.write(f"  Nombre d'images:   {len(morph_images)}\n\n")
        f.write(f"Dossier bona fide:   {bona_fide_dir}\n")
        f.write(f"  Nombre d'images:   {len(bf_images)}\n\n")

        # FIQA Results
        f.write("\n" + "="*80 + "\n")
        f.write("FACE IMAGE QUALITY ASSESSMENT (FIQA)\n")
        f.write("-"*80 + "\n\n")

        f.write("Images Morphees:\n")
        f.write(f"  Moyenne:         {morph_stats['mean']:.6f}\n")
        f.write(f"  Ecart-type:      {morph_stats['std']:.6f}\n")
        f.write(f"  Min:             {morph_stats['min']:.6f}\n")
        f.write(f"  Max:             {morph_stats['max']:.6f}\n")
        f.write(f"  Mediane:         {morph_stats['median']:.6f}\n")
        f.write(f"  Q25:             {morph_stats['q25']:.6f}\n")
        f.write(f"  Q75:             {morph_stats['q75']:.6f}\n")
        f.write(f"  Skewness:        {morph_stats['skewness']:.6f}\n")
        f.write(f"  Kurtosis:        {morph_stats['kurtosis']:.6f}\n")
        f.write(f"  IC 95%%:          [{morph_stats['ci_lower']:.6f}, {morph_stats['ci_upper']:.6f}]\n")
        f.write(f"  Count:           {len(morph_stats['scores'])}\n\n")

        f.write("Images Bona Fide:\n")
        f.write(f"  Moyenne:         {bf_stats['mean']:.6f}\n")
        f.write(f"  Ecart-type:      {bf_stats['std']:.6f}\n")
        f.write(f"  Min:             {bf_stats['min']:.6f}\n")
        f.write(f"  Max:             {bf_stats['max']:.6f}\n")
        f.write(f"  Mediane:         {bf_stats['median']:.6f}\n")
        f.write(f"  Q25:             {bf_stats['q25']:.6f}\n")
        f.write(f"  Q75:             {bf_stats['q75']:.6f}\n")
        f.write(f"  Skewness:        {bf_stats['skewness']:.6f}\n")
        f.write(f"  Kurtosis:        {bf_stats['kurtosis']:.6f}\n")
        f.write(f"  IC 95%%:          [{bf_stats['ci_lower']:.6f}, {bf_stats['ci_upper']:.6f}]\n")
        f.write(f"  Count:           {len(bf_stats['scores'])}\n\n")

        f.write(f"Divergence KL:       {kl_div:.8f}\n")
        f.write(f"Test t (p-value):    {p_value:.8f}\n\n")

        # MAP Results
        f.write("\n" + "="*80 + "\n")
        f.write("MORPHING ATTACK POTENTIAL (MAP)\n")
        f.write("Standard ISO/IEC 20059\n")
        f.write("-"*80 + "\n\n")

        for frs, result in map_results.items():
            f.write(f"{frs.upper()}:\n")
            f.write(f"  MAP Score:       {result['map_score']:.6f}\n")
            f.write(f"  Matches A:       {result['matches_a']}/{result['total_comparisons']}\n")
            f.write(f"  Matches B:       {result['matches_b']}/{result['total_comparisons']}\n")
            f.write(f"  Attack Rate:     {result['map_score']*100:.2f}%\n\n")

        f.write(f"MAP Moyen:           {avg_map:.6f}\n")
        f.write(f"MAP Ecart-type:      {std_map:.6f}\n\n")

        # Algorithmes
        f.write("\n" + "="*80 + "\n")
        f.write("ALGORITHMES ET METHODES UTILISES\n")
        f.write("-"*80 + "\n\n")
        f.write("Optimisation et Analyse Numerique:\n")
        f.write("  - Gradient Descent avec regularisation L2\n")
        f.write("  - Optimisation Convexe (resolution analytique)\n")
        f.write("  - Bootstrap Method (1000 iterations)\n")
        f.write("  - Monte Carlo Estimation\n")
        f.write("  - Principal Component Analysis (PCA)\n")
        f.write("  - Cubic Spline Approximation\n\n")

        f.write("Extraction de Features:\n")
        f.write("  - Sharpness (variance Laplacian)\n")
        f.write("  - Contrast (ecart-type normalise)\n")
        f.write("  - Brightness distribution (mean, std, skew, kurtosis)\n")
        f.write("  - Frequency analysis (FFT magnitude)\n")
        f.write("  - Entropy (contenu informationnel)\n")
        f.write("  - Gradient analysis (Sobel)\n")
        f.write("  - Multi-scale features\n\n")

        f.write("Mesures de Similarite:\n")
        f.write("  - Cosine similarity\n")
        f.write("  - Euclidean distance (normalisee)\n")
        f.write("  - Pearson correlation\n\n")

        f.write("Statistiques Robustes:\n")
        f.write("  - Intervalles de confiance (Bootstrap)\n")
        f.write("  - Moments statistiques (skewness, kurtosis)\n")
        f.write("  - Quartiles (Q25, Q50, Q75)\n")
        f.write("  - Tests d'hypotheses (t-test)\n")
        f.write("  - Kullback-Leibler Divergence\n\n")

        f.write("="*80 + "\n")
        f.write("Fin du rapport\n")
        f.write("="*80 + "\n")

    print("  Rapport sauvegarde avec succes")

    # ===================================================================
    # RESUME FINAL
    # ===================================================================
    print("\n" + "="*80)
    print("ANALYSE TERMINEE AVEC SUCCES")
    print("="*80 + "\n")

    print(f"Tous les resultats ont ete sauvegardes dans: {output_dir}/\n")

    print("Fichiers generes:")
    print("  Visualisations:")
    print("    - fiqa_kde_distribution.png      (Distributions KDE avec IC)")
    print("    - map_comparison.png             (Comparaison MAP multi-systemes)")
    print("    - det_curve.png                  (Courbe DET)")
    print("    - fiqa_boxplot_comparison.png    (Box plots)")
    print("\n  Rapport:")
    print("    - analysis_report.txt            (Rapport complet)")

    print("\n" + "="*80 + "\n")

    return True


def main():
    """Point d'entree principal avec parsing d'arguments"""

    parser = argparse.ArgumentParser(
        description="Analyse statistique de morphings faciaux (version optimisee)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  Analyse de base:
    python analyze_morphs_optimized.py --morph morphing_results --bona-fide sample_data/before_morph

  Avec limite d'images:
    python analyze_morphs_optimized.py --morph morphs --bona-fide originals --max 50

  Dossier de sortie personnalise:
    python analyze_morphs_optimized.py --morph morphs --bona-fide originals --output mes_resultats

  Systemes FRS specifiques:
    python analyze_morphs_optimized.py --morph morphs --bona-fide originals --frs arcface dlib
        """
    )

    parser.add_argument(
        '--morph',
        type=str,
        required=True,
        help='Dossier contenant les images morphees'
    )

    parser.add_argument(
        '--bona-fide',
        type=str,
        required=True,
        help='Dossier contenant les images authentiques (bona fide)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='statistics_output',
        help='Dossier de sortie pour les resultats (defaut: statistics_output)'
    )

    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Limite du nombre d\'images a analyser par categorie'
    )

    parser.add_argument(
        '--frs',
        nargs='+',
        choices=['arcface', 'dlib', 'facenet', 'vggface'],
        default=None,
        help='Systemes FRS a tester (defaut: tous)'
    )

    args = parser.parse_args()

    # Lancement de l'analyse
    success = analyze_face_morphing_dataset(
        morph_dir=args.morph,
        bona_fide_dir=args.bona_fide,
        output_dir=args.output,
        max_images=args.max,
        frs_systems=args.frs
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
