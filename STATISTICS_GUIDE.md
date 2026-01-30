# 📊 Guide d'Utilisation du Système de Statistiques

## Vue d'ensemble

Le système de statistiques implémente les analyses du papier de recherche **SynMorph** (arXiv:2409.05595v1) pour évaluer la qualité et la vulnérabilité des morphings faciaux.

---

## 🎯 Analyses Disponibles

### 1. FIQA - Face Image Quality Assessment

Évalue la qualité biométrique des images faciales selon trois méthodes:

#### Méthode Simple
- Basée sur des métriques classiques de traitement d'images
- **Netteté** : Variance du Laplacian
- **Contraste** : Écart-type des pixels
- **Luminosité** : Moyenne des pixels
- **Bruit** : Évaluation via denoising

Score final : Combinaison pondérée (0-1)

#### FaceQnet v1 (Simulé)
- Approche supervisée end-to-end
- Dans une implémentation complète : modèle CNN pré-entraîné
- Version actuelle : Simulation basée sur métriques + bruit aléatoire

#### SER-FIQ (Simulé)
- Approche non-supervisée basée sur la stabilité
- Dans une implémentation complète : FRS avec dropout
- Version actuelle : Mesure de la variance locale

---

### 2. MAP - Morphing Attack Potential

Mesure l'efficacité des attaques de morphing selon la norme **ISO/IEC 20059**.

#### Calcul
```
MAP = (N_match_A + N_match_B) / (2 × N_total)
```

Où :
- `N_match_A` : Nombre de morphs acceptés comme identité A
- `N_match_B` : Nombre de morphs acceptés comme identité B
- `N_total` : Nombre total de comparaisons

#### Systèmes FRS Testés
1. **ArcFace** : State-of-the-art (SOTA)
2. **Dlib** : Classique basé sur landmarks
3. **Facenet** : Google, basé sur triplet loss
4. **VGGFace** : Oxford, architecture VGG

---

### 3. Visualisations

#### KDE Plots (Kernel Density Estimation)
- Affiche la distribution des scores de qualité
- Compare morphs vs bona fide
- Inclut KL-Divergence pour mesurer la différence

#### DET Curves (Detection Error Tradeoff)
- **MACER** (axe Y) : Morphing Attack Classification Error Rate
- **BPCER** (axe X) : Bona fide Presentation Classification Error Rate
- **EER** : Equal Error Rate (point d'intersection)

#### Box Plots Comparatifs
- Compare les différentes méthodes FIQA
- Visualise médiane, quartiles, outliers

---

## 🚀 Utilisation

### 1. Installation des Dépendances

```bash
pip install numpy opencv-python matplotlib seaborn scipy scikit-learn
```

### 2. Démonstration Rapide

```bash
# Test du module de statistiques
python statistics_module.py
```

Ceci générera des données de démo et produira tous les graphiques.

### 3. Analyse de Vos Morphings

```bash
# Analyser les échantillons de démonstration
python analyze_morphs.py --morph sample_data/after_morph --bona-fide sample_data/before_morph

# Analyser vos résultats de génération
python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph --output my_stats

# Limiter à 50 images pour test rapide
python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph --max 50
```

### 4. Utilisation Programmatique

```python
from statistics_module import FIQAAnalyzer, MAPAnalyzer, StatisticsVisualizer
import cv2

# 1. Charger vos images
morph_images = [cv2.imread(f"morph_{i}.png") for i in range(10)]
bona_fide_images = [cv2.imread(f"original_{i}.png") for i in range(10)]

# 2. Analyse FIQA
fiqa = FIQAAnalyzer()
all_images = morph_images + bona_fide_images
labels = ['morph'] * len(morph_images) + ['bona_fide'] * len(bona_fide_images)
fiqa_stats = fiqa.analyze_dataset(all_images, labels, method='simple')

print(f"Qualité morphs: {fiqa_stats['morph']['mean']:.3f}")
print(f"Qualité originaux: {fiqa_stats['bona_fide']['mean']:.3f}")
print(f"KL-Divergence: {fiqa_stats['kl_divergence']:.4f}")

# 3. Analyse MAP
map_analyzer = MAPAnalyzer()
mated_a = bona_fide_images[:5]
mated_b = bona_fide_images[5:]
map_results = map_analyzer.compute_map(morph_images, mated_a, mated_b, threshold=0.6)

for model, results in map_results.items():
    print(f"{model}: MAP = {results['map_score']:.3f}")

# 4. Visualisations
viz = StatisticsVisualizer(output_dir="my_results")
viz.plot_kde_comparison(fiqa_stats, title="Ma Distribution FIQA")
viz.plot_map_comparison(map_results, title="Mon Analyse MAP")
viz.generate_summary_report(fiqa_stats, map_results)
```

---

## 📄 Fichiers de Sortie

Après l'exécution de `analyze_morphs.py`, vous trouverez dans `statistics_output/` :

| Fichier | Description |
|---------|-------------|
| `fiqa_kde_simple.png` | Distribution KDE - Méthode Simple |
| `fiqa_kde_facequnet.png` | Distribution KDE - FaceQnet v1 |
| `fiqa_kde_serfiq.png` | Distribution KDE - SER-FIQ |
| `map_comparison.png` | Bar chart comparant MAP par FRS |
| `det_curve_fiqa.png` | Courbe DET pour détection |
| `fiqa_methods_comparison.png` | Box plots comparatifs |
| `analysis_report.txt` | Rapport texte complet |

---

## 📊 Interprétation des Résultats

### Scores FIQA
- **> 0.8** : Excellente qualité biométrique
- **0.6 - 0.8** : Bonne qualité
- **0.4 - 0.6** : Qualité moyenne
- **< 0.4** : Faible qualité

### Scores MAP
- **> 0.7** : Attaque très efficace (vulnérabilité élevée)
- **0.5 - 0.7** : Attaque modérément efficace
- **0.3 - 0.5** : Attaque peu efficace
- **< 0.3** : Attaque inefficace

### KL-Divergence
- **Proche de 0** : Distributions très similaires
- **> 0.5** : Distributions différentes
- **> 1.0** : Distributions très différentes

### EER (Equal Error Rate)
- **< 5%** : Excellent détecteur
- **5% - 10%** : Bon détecteur
- **10% - 20%** : Détecteur acceptable
- **> 20%** : Détecteur faible

---

## 🔬 Référence Scientifique

### Papier Original
**SynMorph: Generating Synthetic Face Morphing Dataset with Mated Samples**
- 📄 arXiv:2409.05595v1 [cs.CV] - 9 Septembre 2024
- 👥 Auteurs : Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch
- 🏫 Norwegian University of Science and Technology (NTNU), Darmstadt University

### Standards
- **ISO/IEC 20059** : Biometric presentation attack detection - Part 1: Framework
- **ISO/IEC 19795** : Biometric performance testing and reporting

---

## 💡 Améliorations Futures

Pour une implémentation complète selon SynMorph :

1. **FaceQnet v1** : Intégrer le modèle pré-entraîné
   ```python
   # Télécharger depuis: https://github.com/uam-biometrics/FaceQnet
   ```

2. **SER-FIQ** : Utiliser un vrai FRS avec dropout
   ```python
   # Utiliser ArcFace avec dropout pour stabilité
   ```

3. **Vrais Mated Samples** : Générer avec IFGS/IFGD/FRPCA
   - Nécessite StyleGAN2 et latent editing

4. **Algorithmes MAD** : Entraîner MorphHRNet, Xception, DDFR, LMFD
   - S-MAD : Single image detection
   - D-MAD : Differential detection

---

## 🐛 Dépannage

### Problème : Pas assez d'images
```
❌ ERREUR: Pas assez d'images chargées pour l'analyse!
```
**Solution** : Vérifiez que les répertoires contiennent des images .png/.jpg

### Problème : Module non trouvé
```
ModuleNotFoundError: No module named 'seaborn'
```
**Solution** :
```bash
pip install seaborn scipy
```

### Problème : Mémoire insuffisante
**Solution** : Utilisez `--max` pour limiter les images
```bash
python analyze_morphs.py --max 20
```

---

## 📞 Support

Pour questions ou problèmes :
- Consultez [SYNMORPH_FEATURES.md](SYNMORPH_FEATURES.md)
- Voir le code source : [statistics_module.py](statistics_module.py)
- GitHub Issues : https://github.com/yurri775/FACEMOMO/issues

---

**Auteur** : Marwa
**Projet** : FACEMOMO
**Date** : Janvier 2026
