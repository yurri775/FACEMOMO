# -*- coding: utf-8 -*-
"""
Module de Statistiques pour Face Morphing
Basé sur le papier SynMorph (arXiv:2409.05595v1)

Ce module implémente les analyses statistiques suivantes:
1. Face Image Quality Assessment (FIQA)
2. Morphing Attack Potential (MAP) - ISO/IEC 20059
3. Kernel Density Estimation (KDE) pour visualisation
4. Detection Error Tradeoff (DET) curves
5. Kullback-Leibler Divergence
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


class FIQAAnalyzer:
    """
    Face Image Quality Assessment Analyzer

    Implémente deux méthodes:
    1. FaceQnet v1 - Approche supervisée (simulée pour démo)
    2. SER-FIQ - Approche non-supervisée basée sur stabilité
    """

    def __init__(self):
        self.quality_scores = []
        self.method = "simple"  # simple, facequnet, serfiq

    def compute_quality_score(self, image, method="simple"):
        """
        Calcule le score de qualité d'une image faciale

        Args:
            image: Image BGR (numpy array)
            method: 'simple', 'facequnet', ou 'serfiq'

        Returns:
            float: Score de qualité entre 0 et 1
        """
        if method == "simple":
            return self._compute_simple_quality(image)
        elif method == "facequnet":
            return self._compute_facequnet_quality(image)
        elif method == "serfiq":
            return self._compute_serfiq_quality(image)
        else:
            raise ValueError(f"Méthode inconnue: {method}")

    def _compute_simple_quality(self, image):
        """
        Calcule un score de qualité simple basé sur:
        - Netteté (Laplacian variance)
        - Contraste
        - Luminosité moyenne
        """
        # Convertir en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Netteté (variance du Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(1.0, sharpness / 500.0)  # Normaliser

        # 2. Contraste (écart-type)
        contrast = gray.std()
        contrast_score = min(1.0, contrast / 64.0)

        # 3. Luminosité (vérifier si dans une bonne plage)
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5

        # 4. Score de bruit (inverse)
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        noise_diff = np.abs(gray.astype(float) - noise.astype(float)).mean()
        noise_score = max(0, 1.0 - noise_diff / 20.0)

        # Score combiné
        quality = (
            0.40 * sharpness_score +
            0.25 * contrast_score +
            0.20 * brightness_score +
            0.15 * noise_score
        )

        return float(np.clip(quality, 0, 1))

    def _compute_facequnet_quality(self, image):
        """
        Simule FaceQnet v1 (nécessiterait le modèle pré-entraîné)

        Pour l'instant, utilise une combinaison de métriques
        """
        # Dans une implémentation complète, on chargerait le modèle FaceQnet
        # et on ferait une inférence

        # Pour la démo, on utilise des heuristiques
        simple_score = self._compute_simple_quality(image)

        # Ajouter du bruit gaussien pour simuler la variabilité du modèle
        noise = np.random.normal(0, 0.05)
        score = np.clip(simple_score + noise, 0, 1)

        return float(score)

    def _compute_serfiq_quality(self, image):
        """
        Simule SER-FIQ (nécessiterait un modèle FRS avec dropout)

        SER-FIQ mesure la stabilité des embeddings
        """
        # Dans une implémentation complète, on utiliserait un FRS avec dropout
        # et on mesurerait la variance des embeddings

        # Pour la démo, on combine plusieurs facteurs
        simple_score = self._compute_simple_quality(image)

        # Simuler la stabilité via la cohérence locale de l'image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculer la variance locale (proxy pour la stabilité)
        kernel_size = 5
        local_variance = cv2.blur(gray.astype(float)**2, (kernel_size, kernel_size)) - \
                        cv2.blur(gray.astype(float), (kernel_size, kernel_size))**2
        stability_score = 1.0 - np.clip(local_variance.mean() / 1000.0, 0, 1)

        # Combiner
        score = 0.6 * simple_score + 0.4 * stability_score

        return float(np.clip(score, 0, 1))

    def analyze_dataset(self, images, labels, method="simple"):
        """
        Analyse la qualité d'un dataset entier

        Args:
            images: Liste d'images
            labels: Liste de labels ('morph' ou 'bona_fide')
            method: Méthode FIQA à utiliser

        Returns:
            dict: Statistiques par type d'image
        """
        scores_morph = []
        scores_bona_fide = []

        print(f"📊 Analyse FIQA de {len(images)} images avec méthode '{method}'...")

        for img, label in zip(images, labels):
            score = self.compute_quality_score(img, method=method)

            if label == 'morph':
                scores_morph.append(score)
            else:
                scores_bona_fide.append(score)

        # Calculer les statistiques
        stats_dict = {
            'morph': {
                'scores': scores_morph,
                'mean': np.mean(scores_morph) if scores_morph else 0,
                'std': np.std(scores_morph) if scores_morph else 0,
                'min': np.min(scores_morph) if scores_morph else 0,
                'max': np.max(scores_morph) if scores_morph else 0,
                'median': np.median(scores_morph) if scores_morph else 0,
            },
            'bona_fide': {
                'scores': scores_bona_fide,
                'mean': np.mean(scores_bona_fide) if scores_bona_fide else 0,
                'std': np.std(scores_bona_fide) if scores_bona_fide else 0,
                'min': np.min(scores_bona_fide) if scores_bona_fide else 0,
                'max': np.max(scores_bona_fide) if scores_bona_fide else 0,
                'median': np.median(scores_bona_fide) if scores_bona_fide else 0,
            }
        }

        # Calculer KL-Divergence
        if scores_morph and scores_bona_fide:
            kl_div = self._compute_kl_divergence(scores_morph, scores_bona_fide)
            stats_dict['kl_divergence'] = kl_div

        return stats_dict


    def _compute_kl_divergence(self, scores1, scores2, bins=50):
        """
        Calcule la divergence KL entre deux distributions
        """
        # Créer des histogrammes normalisés
        hist1, bin_edges = np.histogram(scores1, bins=bins, range=(0, 1), density=True)
        hist2, _ = np.histogram(scores2, bins=bins, range=(0, 1), density=True)

        # Normaliser pour obtenir des distributions de probabilité
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # Éviter les divisions par zéro
        hist1 = np.clip(hist1, 1e-10, None)
        hist2 = np.clip(hist2, 1e-10, None)

        # Calculer KL divergence
        kl_div = np.sum(hist1 * np.log(hist1 / hist2))

        return float(kl_div)


class MAPAnalyzer:
    """
    Morphing Attack Potential Analyzer

    Implémente le calcul du MAP selon ISO/IEC 20059
    """

    def __init__(self, frs_models=None):
        """
        Args:
            frs_models: Liste des modèles FRS à utiliser
                       (pour démo, on simule avec des embeddings aléatoires)
        """
        self.frs_models = frs_models or ['ArcFace', 'Dlib', 'Facenet', 'VGGFace']

    def extract_embedding(self, image, model='simple'):
        """
        Extrait l'embedding facial d'une image

        Dans une implémentation complète, on utiliserait un vrai modèle FRS
        Pour la démo, on simule avec des features basiques
        """
        # Redimensionner à une taille standard
        img_resized = cv2.resize(image, (128, 128))

        if len(img_resized.shape) == 3:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_resized

        # Simuler un embedding avec HOG ou autre descripteur
        # Dans la vraie implémentation, utiliser ArcFace, etc.

        # Pour démo: utiliser des statistiques simples comme "embedding"
        embedding = np.array([
            gray.mean(),
            gray.std(),
            gray.min(),
            gray.max(),
            np.median(gray),
            cv2.Laplacian(gray, cv2.CV_64F).var(),
        ])

        # Normaliser
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        return embedding

    def compute_similarity(self, emb1, emb2, metric='cosine'):
        """
        Calcule la similarité entre deux embeddings

        Args:
            emb1, emb2: Embeddings à comparer
            metric: 'cosine' ou 'euclidean'

        Returns:
            float: Score de similarité (plus proche de 1 = plus similaire)
        """
        if metric == 'cosine':
            # Similarité cosinus (1 - distance cosinus)
            return 1.0 - cosine(emb1, emb2)
        elif metric == 'euclidean':
            # Distance euclidienne normalisée
            dist = np.linalg.norm(emb1 - emb2)
            return 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Métrique inconnue: {metric}")

    def compute_map(self, morph_images, mated_samples_a, mated_samples_b, threshold=0.7):
        """
        Calcule le Morphing Attack Potential (MAP)

        MAP = (N_match_A + N_match_B) / (2 * N_total)

        Args:
            morph_images: Liste d'images morphées
            mated_samples_a: Liste de samples de la personne A
            mated_samples_b: Liste de samples de la personne B
            threshold: Seuil de similarité pour acceptation

        Returns:
            dict: Résultats MAP par modèle FRS
        """
        results = {}

        print(f"\n📈 Calcul du Morphing Attack Potential (MAP)...")
        print(f"   Morphs: {len(morph_images)}, Mated A: {len(mated_samples_a)}, Mated B: {len(mated_samples_b)}")

        for model_name in self.frs_models:
            print(f"   • Modèle {model_name}...", end=" ")

            match_count_a = 0
            match_count_b = 0
            total_comparisons = 0

            for morph_img in morph_images:
                # Extraire embedding du morph
                morph_emb = self.extract_embedding(morph_img, model=model_name)

                # Comparer avec mated samples A
                for mated_img_a in mated_samples_a:
                    mated_emb_a = self.extract_embedding(mated_img_a, model=model_name)
                    similarity_a = self.compute_similarity(morph_emb, mated_emb_a)

                    if similarity_a >= threshold:
                        match_count_a += 1
                    total_comparisons += 1

                # Comparer avec mated samples B
                for mated_img_b in mated_samples_b:
                    mated_emb_b = self.extract_embedding(mated_img_b, model=model_name)
                    similarity_b = self.compute_similarity(morph_emb, mated_emb_b)

                    if similarity_b >= threshold:
                        match_count_b += 1

            # Calculer MAP
            if total_comparisons > 0:
                map_score = (match_count_a + match_count_b) / (2.0 * len(morph_images) * max(len(mated_samples_a), len(mated_samples_b)))
            else:
                map_score = 0.0

            results[model_name] = {
                'map_score': map_score,
                'match_a': match_count_a,
                'match_b': match_count_b,
                'total': total_comparisons
            }

            print(f"MAP = {map_score:.3f}")

        return results


class StatisticsVisualizer:
    """
    Visualiseur de statistiques pour morphing
    """

    def __init__(self, output_dir="./statistics_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Style Seaborn
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'

    def plot_kde_comparison(self, scores_dict, title="Distribution FIQA", filename="fiqa_kde.png"):
        """
        Crée un KDE plot pour comparer les distributions de qualité

        Args:
            scores_dict: Dict avec 'morph' et 'bona_fide' scores
            title: Titre du graphique
            filename: Nom du fichier de sortie
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # KDE pour morphs
        if scores_dict['morph']['scores']:
            sns.kdeplot(data=scores_dict['morph']['scores'],
                       label='Morphed Images',
                       color='red',
                       linewidth=2.5,
                       ax=ax)

        # KDE pour bona fide
        if scores_dict['bona_fide']['scores']:
            sns.kdeplot(data=scores_dict['bona_fide']['scores'],
                       label='Bona Fide Images',
                       color='blue',
                       linewidth=2.5,
                       ax=ax)

        ax.set_xlabel('Quality Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Ajouter statistiques
        stats_text = f"""
Morphed Images:
  Mean: {scores_dict['morph']['mean']:.3f}
  Std: {scores_dict['morph']['std']:.3f}

Bona Fide Images:
  Mean: {scores_dict['bona_fide']['mean']:.3f}
  Std: {scores_dict['bona_fide']['std']:.3f}
"""
        if 'kl_divergence' in scores_dict:
            stats_text += f"\nKL-Divergence: {scores_dict['kl_divergence']:.4f}"

        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ KDE plot sauvegardé: {output_path}")
        return output_path

    def plot_map_comparison(self, map_results, title="Morphing Attack Potential", filename="map_comparison.png"):
        """
        Visualise les résultats MAP pour différents FRS

        Args:
            map_results: Résultats du MAPAnalyzer
            title: Titre du graphique
            filename: Nom du fichier
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(map_results.keys())
        map_scores = [map_results[m]['map_score'] for m in models]

        # Bar plot
        bars = ax.bar(models, map_scores, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.7)

        # Ajouter valeurs sur les barres
        for bar, score in zip(bars, map_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('MAP Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Face Recognition System', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ MAP comparison sauvegardé: {output_path}")
        return output_path

    def plot_det_curve(self, y_true, y_scores, title="DET Curve", filename="det_curve.png"):
        """
        Crée une courbe DET (Detection Error Tradeoff)

        Args:
            y_true: Labels vrais (0 = bona fide, 1 = morph)
            y_scores: Scores de détection
            title: Titre
            filename: Nom du fichier
        """
        # Calculer FPR et FNR
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot DET curve
        ax.plot(fpr * 100, fnr * 100, 'b-', linewidth=2.5, label='DET Curve')

        # Point EER (Equal Error Rate)
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        ax.plot(fpr[eer_idx] * 100, fnr[eer_idx] * 100, 'ro',
               markersize=10, label=f'EER = {eer*100:.2f}%')

        ax.set_xlabel('BPCER (%) - Bona Fide Presentation Classification Error Rate',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('MACER (%) - Morphing Attack Classification Error Rate',
                     fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)

        # Diagonal line
        ax.plot([0, 50], [0, 50], 'k--', alpha=0.3, label='Random Classifier')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ DET curve sauvegardée: {output_path}")
        return output_path

    def generate_summary_report(self, fiqa_stats, map_results, output_file="summary_report.txt"):
        """
        Génère un rapport texte résumant toutes les statistiques
        """
        report_path = self.output_dir / output_file

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" "*20 + "RAPPORT D'ANALYSE STATISTIQUE\n")
            f.write(" "*15 + "Face Morphing - Basé sur SynMorph\n")
            f.write("="*80 + "\n\n")

            # Section FIQA
            f.write("📊 FACE IMAGE QUALITY ASSESSMENT (FIQA)\n")
            f.write("-"*80 + "\n\n")

            f.write("Morphed Images:\n")
            f.write(f"  • Moyenne:  {fiqa_stats['morph']['mean']:.4f}\n")
            f.write(f"  • Écart-type: {fiqa_stats['morph']['std']:.4f}\n")
            f.write(f"  • Min:      {fiqa_stats['morph']['min']:.4f}\n")
            f.write(f"  • Max:      {fiqa_stats['morph']['max']:.4f}\n")
            f.write(f"  • Médiane:  {fiqa_stats['morph']['median']:.4f}\n")
            f.write(f"  • Count:    {len(fiqa_stats['morph']['scores'])}\n\n")

            f.write("Bona Fide Images:\n")
            f.write(f"  • Moyenne:  {fiqa_stats['bona_fide']['mean']:.4f}\n")
            f.write(f"  • Écart-type: {fiqa_stats['bona_fide']['std']:.4f}\n")
            f.write(f"  • Min:      {fiqa_stats['bona_fide']['min']:.4f}\n")
            f.write(f"  • Max:      {fiqa_stats['bona_fide']['max']:.4f}\n")
            f.write(f"  • Médiane:  {fiqa_stats['bona_fide']['median']:.4f}\n")
            f.write(f"  • Count:    {len(fiqa_stats['bona_fide']['scores'])}\n\n")

            if 'kl_divergence' in fiqa_stats:
                f.write(f"KL-Divergence: {fiqa_stats['kl_divergence']:.6f}\n\n")

            # Section MAP
            f.write("\n" + "="*80 + "\n")
            f.write("📈 MORPHING ATTACK POTENTIAL (MAP)\n")
            f.write("-"*80 + "\n\n")

            for model_name, results in map_results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  • MAP Score:    {results['map_score']:.4f}\n")
                f.write(f"  • Matches A:    {results['match_a']}\n")
                f.write(f"  • Matches B:    {results['match_b']}\n")
                f.write(f"  • Total Comps:  {results['total']}\n\n")

            # Moyenne MAP
            avg_map = np.mean([r['map_score'] for r in map_results.values()])
            f.write(f"MAP Moyen: {avg_map:.4f}\n\n")

            f.write("="*80 + "\n")
            f.write("Fin du rapport\n")
            f.write("="*80 + "\n")

        print(f"   ✓ Rapport sauvegardé: {report_path}")
        return report_path


def demo_usage():
    """
    Démontre l'utilisation du module de statistiques
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                MODULE DE STATISTIQUES - DÉMONSTRATION                 ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Créer des données de démo
    print("\n1️⃣  Création de données de démonstration...")

    # Simuler quelques images
    morph_images = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(20)]
    bona_fide_images = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(20)]

    # Labels
    images = morph_images + bona_fide_images
    labels = ['morph'] * len(morph_images) + ['bona_fide'] * len(bona_fide_images)

    # 2. FIQA Analysis
    print("\n2️⃣  Analyse FIQA...")
    fiqa = FIQAAnalyzer()
    fiqa_stats = fiqa.analyze_dataset(images, labels, method="simple")

    print(f"\n   Résultats FIQA:")
    print(f"   • Morph quality:     {fiqa_stats['morph']['mean']:.3f} ± {fiqa_stats['morph']['std']:.3f}")
    print(f"   • Bona fide quality: {fiqa_stats['bona_fide']['mean']:.3f} ± {fiqa_stats['bona_fide']['std']:.3f}")
    print(f"   • KL-Divergence:     {fiqa_stats.get('kl_divergence', 0):.4f}")

    # 3. MAP Analysis
    print("\n3️⃣  Analyse MAP...")
    map_analyzer = MAPAnalyzer()

    # Simuler des mated samples
    mated_a = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]
    mated_b = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]

    map_results = map_analyzer.compute_map(morph_images[:5], mated_a, mated_b)

    # 4. Visualisations
    print("\n4️⃣  Génération des visualisations...")
    viz = StatisticsVisualizer()

    viz.plot_kde_comparison(fiqa_stats, title="FIQA - Démo")
    viz.plot_map_comparison(map_results, title="MAP - Démo")

    # DET curve (simulée)
    y_true = np.array([0]*20 + [1]*20)
    y_scores = np.random.rand(40)
    viz.plot_det_curve(y_true, y_scores, title="DET Curve - Démo")

    # 5. Rapport
    print("\n5️⃣  Génération du rapport...")
    viz.generate_summary_report(fiqa_stats, map_results)

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                      DÉMONSTRATION TERMINÉE ✅                        ║
║                                                                       ║
║  Les résultats ont été sauvegardés dans ./statistics_output/         ║
╚═══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    demo_usage()
