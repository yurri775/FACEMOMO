# -*- coding: utf-8 -*-
"""
Démonstration Rapide du Système de Statistiques
Execute ce script pour voir un exemple d'utilisation complet
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
from statistics_module import FIQAAnalyzer, MAPAnalyzer, StatisticsVisualizer


def create_demo_morphs():
    """Crée quelques morphings de démonstration"""
    print("\n🎨 Création de morphings de démonstration...")

    # Créer des images synthétiques pour la démo
    morphs = []
    originals = []

    for i in range(5):
        # Image originale simulée
        img_orig = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)

        # Ajouter du bruit pour simuler une texture faciale
        noise = np.random.normal(0, 10, img_orig.shape).astype(np.uint8)
        img_orig = cv2.add(img_orig, noise)

        # Morphing simulé (légèrement plus flou)
        img_morph = cv2.GaussianBlur(img_orig, (5, 5), 0)
        img_morph = cv2.addWeighted(img_orig, 0.7, img_morph, 0.3, 0)

        morphs.append(img_morph)
        originals.append(img_orig)

    print(f"   ✓ {len(morphs)} morphs créés")
    print(f"   ✓ {len(originals)} images originales créées")

    return morphs, originals


def main():
    """Démonstration complète"""

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║            DÉMONSTRATION RAPIDE - SYSTÈME DE STATISTIQUES             ║
║                    Basé sur SynMorph (arXiv:2409.05595)              ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # 1. Créer des données de démonstration
    morphs, originals = create_demo_morphs()

    # Combiner les données
    all_images = morphs + originals
    labels = ['morph'] * len(morphs) + ['bona_fide'] * len(originals)

    # 2. Analyse FIQA
    print("\n" + "="*80)
    print("1️⃣  ANALYSE FIQA (Face Image Quality Assessment)")
    print("="*80)

    fiqa = FIQAAnalyzer()

    print("\n📊 Méthode Simple (Laplacian + Contraste + Luminosité)...")
    stats_simple = fiqa.analyze_dataset(all_images, labels, method='simple')

    print(f"""
Résultats:
╭─────────────────────────────────────────────╮
│ Morphs:                                     │
│   • Qualité moyenne:  {stats_simple['morph']['mean']:.4f}            │
│   • Écart-type:       {stats_simple['morph']['std']:.4f}            │
│                                             │
│ Originaux:                                  │
│   • Qualité moyenne:  {stats_simple['bona_fide']['mean']:.4f}            │
│   • Écart-type:       {stats_simple['bona_fide']['std']:.4f}            │
│                                             │
│ KL-Divergence:        {stats_simple.get('kl_divergence', 0):.6f}            │
╰─────────────────────────────────────────────╯
""")

    print("📊 Méthode FaceQnet v1 (Simulée)...")
    stats_facequnet = fiqa.analyze_dataset(all_images, labels, method='facequnet')

    print(f"   • Qualité morphs:     {stats_facequnet['morph']['mean']:.4f}")
    print(f"   • Qualité originaux:  {stats_facequnet['bona_fide']['mean']:.4f}")

    print("\n📊 Méthode SER-FIQ (Simulée)...")
    stats_serfiq = fiqa.analyze_dataset(all_images, labels, method='serfiq')

    print(f"   • Qualité morphs:     {stats_serfiq['morph']['mean']:.4f}")
    print(f"   • Qualité originaux:  {stats_serfiq['bona_fide']['mean']:.4f}")

    # 3. Analyse MAP
    print("\n" + "="*80)
    print("2️⃣  ANALYSE MAP (Morphing Attack Potential)")
    print("="*80)

    map_analyzer = MAPAnalyzer(frs_models=['ArcFace', 'Dlib', 'Facenet', 'VGGFace'])

    # Diviser les originaux en deux groupes (simule deux identités)
    mated_a = originals[:len(originals)//2]
    mated_b = originals[len(originals)//2:]

    print(f"\n📈 Calcul MAP avec {len(morphs)} morphs...")
    print(f"   • Mated samples A: {len(mated_a)}")
    print(f"   • Mated samples B: {len(mated_b)}")

    map_results = map_analyzer.compute_map(morphs, mated_a, mated_b, threshold=0.6)

    print(f"""
Résultats MAP:
╭─────────────────────────────────────────────╮""")

    for model, results in map_results.items():
        print(f"""│ {model:15s}                         │
│   • MAP Score: {results['map_score']:6.4f}                   │""")

    avg_map = np.mean([r['map_score'] for r in map_results.values()])
    print(f"""├─────────────────────────────────────────────┤
│ MAP Moyen:     {avg_map:6.4f}                   │
╰─────────────────────────────────────────────╯
""")

    # 4. Visualisations
    print("="*80)
    print("3️⃣  GÉNÉRATION DES VISUALISATIONS")
    print("="*80)

    output_dir = Path("demo_statistics_output")
    output_dir.mkdir(exist_ok=True)

    viz = StatisticsVisualizer(output_dir=str(output_dir))

    print("\n📊 Création des graphiques...")

    # KDE plots
    viz.plot_kde_comparison(
        stats_simple,
        title="Distribution FIQA - Démonstration Rapide",
        filename="demo_fiqa_kde.png"
    )

    # MAP comparison
    viz.plot_map_comparison(
        map_results,
        title="MAP par FRS - Démonstration",
        filename="demo_map_comparison.png"
    )

    # DET curve
    y_true = np.array([1] * len(morphs) + [0] * len(originals))
    y_scores = np.array(
        stats_simple['morph']['scores'] +
        stats_simple['bona_fide']['scores']
    )

    viz.plot_det_curve(
        y_true,
        y_scores,
        title="DET Curve - Démonstration",
        filename="demo_det_curve.png"
    )

    # Rapport
    print("\n📄 Génération du rapport...")
    viz.generate_summary_report(
        stats_simple,
        map_results,
        output_file="demo_report.txt"
    )

    # Résumé final
    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                      DÉMONSTRATION TERMINÉE ! ✅                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║ 📊 Analyses effectuées:                                              ║
║   ✓ FIQA - 3 méthodes                                                ║
║   ✓ MAP - 4 systèmes FRS                                             ║
║   ✓ Visualisations KDE, DET, comparaisons                            ║
║   ✓ Rapport texte complet                                            ║
║                                                                       ║
║ 📁 Résultats sauvegardés dans: {str(output_dir):33s} ║
║                                                                       ║
║ 📄 Fichiers générés:                                                 ║
║   • demo_fiqa_kde.png          - Distribution de qualité             ║
║   • demo_map_comparison.png    - Comparaison MAP                     ║
║   • demo_det_curve.png         - Courbe DET                          ║
║   • demo_report.txt            - Rapport complet                     ║
║                                                                       ║
║ 💡 Pour analyser vos vraies données:                                 ║
║    python analyze_morphs.py --morph <dir> --bona-fide <dir>          ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
