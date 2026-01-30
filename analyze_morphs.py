# -*- coding: utf-8 -*-
"""
Script d'Analyse Statistique des Morphings Faciaux
Basé sur le papier SynMorph (arXiv:2409.05595v1)

Ce script:
1. Charge les images morphées et non-morphées
2. Effectue l'analyse FIQA (Face Image Quality Assessment)
3. Calcule le MAP (Morphing Attack Potential)
4. Génère les visualisations (KDE plots, DET curves)
5. Produit un rapport complet
"""

import sys
import io
import os

# Configuration de l'encodage UTF-8 pour Windows
if sys.platform == 'win32':
    # Redéfinir stdout et stderr pour supporter UTF-8
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
from statistics_module import FIQAAnalyzer, MAPAnalyzer, StatisticsVisualizer


def load_images_from_directory(directory, max_images=None):
    """
    Charge toutes les images d'un répertoire

    Args:
        directory: Chemin du répertoire
        max_images: Nombre maximum d'images à charger (None = toutes)

    Returns:
        list: Liste d'images chargées
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"⚠️  Répertoire inexistant: {directory}")
        return []

    # Extensions supportées
    extensions = ['.png', '.jpg', '.jpeg', '.bmp']

    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))

    # Limiter si nécessaire
    if max_images:
        image_files = image_files[:max_images]

    print(f"   📁 {len(image_files)} images trouvées dans {directory.name}")

    # Charger les images
    images = []
    for img_file in tqdm(image_files, desc=f"   Chargement {directory.name}", ncols=80):
        try:
            img = cv2.imread(str(img_file))
            if img is not None:
                images.append(img)
        except Exception as e:
            print(f"   ⚠️  Erreur lecture {img_file.name}: {e}")

    return images


def analyze_morphing_dataset(morph_dir, bona_fide_dir, output_dir="./statistics_output", max_images=None):
    """
    Analyse complète d'un dataset de morphing

    Args:
        morph_dir: Répertoire contenant les images morphées
        bona_fide_dir: Répertoire contenant les images non-morphées
        output_dir: Répertoire de sortie pour les résultats
        max_images: Nombre max d'images par catégorie
    """

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║          ANALYSE STATISTIQUE DES MORPHINGS FACIAUX                    ║
║              Basé sur le papier SynMorph                              ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Créer le dossier de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # ==================== CHARGEMENT DES DONNÉES ====================
    print("\n1️⃣  CHARGEMENT DES DONNÉES")
    print("="*80)

    print("\n📥 Chargement des images morphées...")
    morph_images = load_images_from_directory(morph_dir, max_images)

    print("\n📥 Chargement des images bona fide...")
    bona_fide_images = load_images_from_directory(bona_fide_dir, max_images)

    if len(morph_images) == 0 or len(bona_fide_images) == 0:
        print("\n❌ ERREUR: Pas assez d'images chargées pour l'analyse!")
        print(f"   • Morphs: {len(morph_images)}")
        print(f"   • Bona fide: {len(bona_fide_images)}")
        return

    print(f"""
✅ Données chargées avec succès:
   • Images morphées:     {len(morph_images)}
   • Images bona fide:    {len(bona_fide_images)}
   • Total:               {len(morph_images) + len(bona_fide_images)}
""")

    # Combiner les images et labels
    all_images = morph_images + bona_fide_images
    all_labels = ['morph'] * len(morph_images) + ['bona_fide'] * len(bona_fide_images)

    # ==================== ANALYSE FIQA ====================
    print("\n2️⃣  ANALYSE FIQA (Face Image Quality Assessment)")
    print("="*80)

    fiqa_analyzer = FIQAAnalyzer()

    # Tester avec différentes méthodes
    fiqa_methods = {
        'simple': 'Simple (Laplacian + Contrast)',
        'facequnet': 'FaceQnet v1 (Simulé)',
        'serfiq': 'SER-FIQ (Simulé)'
    }

    fiqa_results = {}

    for method, description in fiqa_methods.items():
        print(f"\n📊 Méthode: {description}")
        fiqa_stats = fiqa_analyzer.analyze_dataset(all_images, all_labels, method=method)
        fiqa_results[method] = fiqa_stats

        # Afficher résumé
        print(f"""
   Résultats:
   ┌─────────────────────────────────────────────────────┐
   │ Morphed Images:                                     │
   │   • Moyenne:  {fiqa_stats['morph']['mean']:6.4f}                           │
   │   • Std:      {fiqa_stats['morph']['std']:6.4f}                           │
   │   • Min:      {fiqa_stats['morph']['min']:6.4f}                           │
   │   • Max:      {fiqa_stats['morph']['max']:6.4f}                           │
   │   • Médiane:  {fiqa_stats['morph']['median']:6.4f}                           │
   ├─────────────────────────────────────────────────────┤
   │ Bona Fide Images:                                   │
   │   • Moyenne:  {fiqa_stats['bona_fide']['mean']:6.4f}                           │
   │   • Std:      {fiqa_stats['bona_fide']['std']:6.4f}                           │
   │   • Min:      {fiqa_stats['bona_fide']['min']:6.4f}                           │
   │   • Max:      {fiqa_stats['bona_fide']['max']:6.4f}                           │
   │   • Médiane:  {fiqa_stats['bona_fide']['median']:6.4f}                           │
   ├─────────────────────────────────────────────────────┤
   │ KL-Divergence: {fiqa_stats.get('kl_divergence', 0):6.4f}                        │
   └─────────────────────────────────────────────────────┘
""")

    # ==================== ANALYSE MAP ====================
    print("\n3️⃣  ANALYSE MAP (Morphing Attack Potential)")
    print("="*80)

    # Pour MAP, on a besoin de mated samples
    # On va utiliser un sous-ensemble des images bona fide comme proxy
    print("\n📈 Calcul du MAP avec simulation de mated samples...")

    # Diviser bona fide en deux groupes (simule deux identités)
    mid_point = len(bona_fide_images) // 2
    mated_samples_a = bona_fide_images[:mid_point]
    mated_samples_b = bona_fide_images[mid_point:]

    # Limiter pour accélérer
    max_morphs_map = min(10, len(morph_images))
    max_mated = min(5, len(mated_samples_a), len(mated_samples_b))

    print(f"   • Morphs à analyser:  {max_morphs_map}")
    print(f"   • Mated samples A:    {max_mated}")
    print(f"   • Mated samples B:    {max_mated}")

    map_analyzer = MAPAnalyzer(frs_models=['ArcFace', 'Dlib', 'Facenet', 'VGGFace'])
    map_results = map_analyzer.compute_map(
        morph_images[:max_morphs_map],
        mated_samples_a[:max_mated],
        mated_samples_b[:max_mated],
        threshold=0.6
    )

    # Afficher résultats MAP
    print(f"""
   Résultats MAP:
   ┌─────────────────────────────────────────────────────┐""")

    for model_name, results in map_results.items():
        print(f"""   │ {model_name:15s}                                  │
   │   • MAP Score:    {results['map_score']:6.4f}                       │
   │   • Matches A:    {results['match_a']:6d}                       │
   │   • Matches B:    {results['match_b']:6d}                       │""")

    avg_map = np.mean([r['map_score'] for r in map_results.values()])
    print(f"""   ├─────────────────────────────────────────────────────┤
   │ MAP Moyen:        {avg_map:6.4f}                       │
   └─────────────────────────────────────────────────────┘
""")

    # ==================== VISUALISATIONS ====================
    print("\n4️⃣  GÉNÉRATION DES VISUALISATIONS")
    print("="*80)

    viz = StatisticsVisualizer(output_dir=output_dir)

    # KDE plots pour chaque méthode FIQA
    print("\n📊 Création des KDE plots...")
    for method, stats in fiqa_results.items():
        viz.plot_kde_comparison(
            stats,
            title=f"Distribution FIQA - {fiqa_methods[method]}",
            filename=f"fiqa_kde_{method}.png"
        )

    # MAP comparison
    print("\n📈 Création du graphique MAP...")
    viz.plot_map_comparison(
        map_results,
        title="Morphing Attack Potential par FRS",
        filename="map_comparison.png"
    )

    # DET Curve (simulée avec scores FIQA)
    print("\n📉 Création de la courbe DET...")

    # Utiliser les scores FIQA comme proxy pour la détection
    y_true = np.array([1] * len(morph_images) + [0] * len(bona_fide_images))
    y_scores = np.array(
        fiqa_results['simple']['morph']['scores'] +
        fiqa_results['simple']['bona_fide']['scores']
    )

    viz.plot_det_curve(
        y_true,
        y_scores,
        title="DET Curve - Détection basée sur FIQA Simple",
        filename="det_curve_fiqa.png"
    )

    # ==================== RAPPORT FINAL ====================
    print("\n5️⃣  GÉNÉRATION DU RAPPORT FINAL")
    print("="*80)

    viz.generate_summary_report(
        fiqa_results['simple'],  # Utiliser les résultats simples
        map_results,
        output_file="analysis_report.txt"
    )

    # ==================== GRAPHIQUES COMPARATIFS ====================
    print("\n6️⃣  GRAPHIQUES COMPARATIFS")
    print("="*80)

    # Comparaison des méthodes FIQA
    print("\n📊 Création du graphique comparatif FIQA...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Comparaison des Méthodes FIQA', fontsize=16, fontweight='bold')

    for idx, (method, stats) in enumerate(fiqa_results.items()):
        ax = axes[idx]

        # Box plots
        data = [stats['morph']['scores'], stats['bona_fide']['scores']]
        bp = ax.boxplot(data, labels=['Morphed', 'Bona Fide'],
                       patch_artist=True)

        # Couleurs
        colors = ['#e74c3c', '#3498db']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Quality Score', fontsize=11)
        ax.set_title(fiqa_methods[method], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    comparison_path = output_path / "fiqa_methods_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Comparaison FIQA sauvegardée: {comparison_path}")

    # ==================== RÉSUMÉ FINAL ====================
    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                      ANALYSE TERMINÉE ! ✅                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║ 📊 Analyses effectuées:                                              ║
║   • FIQA (3 méthodes)                                                ║
║   • MAP (4 FRS)                                                      ║
║   • Visualisations (KDE, DET, comparaisons)                          ║
║                                                                       ║
║ 📁 Résultats sauvegardés dans: {str(output_path):33s} ║
║                                                                       ║
║ 📄 Fichiers générés:                                                 ║
║   • fiqa_kde_*.png         - Distributions de qualité                ║
║   • map_comparison.png     - Comparaison MAP                         ║
║   • det_curve_fiqa.png     - Courbe DET                              ║
║   • fiqa_methods_comparison.png - Comparaison méthodes FIQA          ║
║   • analysis_report.txt    - Rapport texte complet                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    return fiqa_results, map_results


def main():
    """
    Fonction principale avec arguments en ligne de commande
    """

    parser = argparse.ArgumentParser(
        description='Analyse statistique des morphings faciaux (SynMorph)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Analyser les échantillons de démonstration
  python analyze_morphs.py --morph sample_data/after_morph --bona-fide sample_data/before_morph

  # Analyser les résultats de génération
  python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph

  # Limiter le nombre d'images
  python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph --max 50
        """
    )

    parser.add_argument(
        '--morph',
        type=str,
        default='sample_data/after_morph',
        help='Répertoire contenant les images morphées (défaut: sample_data/after_morph)'
    )

    parser.add_argument(
        '--bona-fide',
        type=str,
        default='sample_data/before_morph',
        help='Répertoire contenant les images bona fide (défaut: sample_data/before_morph)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='statistics_output',
        help='Répertoire de sortie pour les résultats (défaut: statistics_output)'
    )

    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Nombre maximum d\'images par catégorie (défaut: toutes)'
    )

    args = parser.parse_args()

    # Lancer l'analyse
    analyze_morphing_dataset(
        morph_dir=args.morph,
        bona_fide_dir=args.bona_fide,
        output_dir=args.output,
        max_images=args.max
    )


if __name__ == "__main__":
    main()
