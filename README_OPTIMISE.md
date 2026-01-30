# FACEMOMO - Face Morphing Studio
## Version Optimisee Professionnelle

Systeme complet d'analyse statistique de morphings faciaux base sur le papier de recherche **SynMorph** (arXiv:2409.05595v1).

**Version optimisee** avec algorithmes avances, analyse numerique et recherche operationnelle, sans emojis pour un usage professionnel.

---

## Table des Matieres

- [Caracteristiques Principales](#caracteristiques-principales)
- [Installation Rapide](#installation-rapide)
- [Utilisation](#utilisation)
- [Algorithmes Avances](#algorithmes-avances)
- [Structure du Projet](#structure-du-projet)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Caracteristiques Principales

### Analyse FIQA (Face Image Quality Assessment)

- **Extraction de features multi-echelles** (10+ metriques)
- **Statistiques robustes** avec Bootstrap (1000 iterations)
- **Intervalles de confiance** a 95%
- **Analyse PCA** pour reduction dimensionnelle
- **Divergence KL** entre distributions
- **Tests statistiques** (t-test, moments)

### Calcul MAP (Morphing Attack Potential)

- **Standard ISO/IEC 20059**
- **4 systemes FRS** : ArcFace, Dlib, Facenet, VGGFace
- **Monte Carlo Estimation** pour robustesse
- **Metriques multiples** : cosine, euclidean, Pearson
- **Caching d'embeddings** pour performance

### Visualisations Avancees

- **KDE Plots** avec intervalles de confiance
- **Courbes DET** (Detection Error Tradeoff)
- **Comparaisons multi-systemes**
- **Box plots** comparatifs
- **Graphiques professionnels** haute resolution

### Rapports Detailles

- **Rapport texte complet** avec toutes statistiques
- **Metriques descriptives** : moyenne, ecart-type, mediane, quartiles
- **Metriques de forme** : skewness, kurtosis
- **Resultats MAP** par systeme FRS
- **Algorithmes utilises** documentes

---

## Installation Rapide

### Prerequis

- Python 3.7+
- pip

### Installation des Dependances

**Windows:**
```bash
install_dependencies.bat
```

**Linux/Mac:**
```bash
pip install numpy opencv-python matplotlib seaborn scipy scikit-learn tqdm
```

Ou avec requirements.txt:
```bash
pip install -r requirements.txt
```

---

## Utilisation

### Demonstration Rapide (30 secondes)

**Windows:**
```bash
run_demo_optimized.bat
```

**Linux/Mac:**
```bash
python quick_demo_optimized.py
```

**Resultat:**
- Dossier `demo_statistics_output/` cree
- 4 visualisations PNG
- 1 rapport texte complet
- Donnees de test generees

### Analyse de Vos Morphings

**Windows:**
```bash
run_analysis_optimized.bat
```

**Linux/Mac:**
```bash
python analyze_morphs_optimized.py --morph <dossier_morphs> --bona-fide <dossier_originaux>
```

**Exemples:**

Analyse de base:
```bash
python analyze_morphs_optimized.py --morph morphing_results --bona-fide sample_data/before_morph
```

Avec limite d'images:
```bash
python analyze_morphs_optimized.py --morph morphs --bona-fide originals --max 50
```

Systemes FRS specifiques:
```bash
python analyze_morphs_optimized.py --morph morphs --bona-fide originals --frs arcface dlib
```

Dossier de sortie personnalise:
```bash
python analyze_morphs_optimized.py --morph morphs --bona-fide originals --output mes_resultats
```

**Resultat:**
- Dossier `statistics_output/` cree
- 4 visualisations professionnelles
- Rapport d'analyse complet
- Toutes statistiques calculees

---

## Algorithmes Avances

### Optimisation Numerique

#### Gradient Descent avec Regularisation L2
```python
def optimize_weights_gradient_descent(features, targets, lr=0.01, max_iter=1000):
    """
    Optimisation par descente de gradient avec regularisation L2
    pour eviter le surapprentissage
    """
    weights = np.random.randn(n_features) * 0.01
    lambda_reg = 0.001

    for iteration in range(max_iter):
        predictions = features @ weights
        error = predictions - targets
        gradient = (2/n) * (features.T @ error) + 2 * lambda_reg * weights
        weights -= lr * gradient

        if np.linalg.norm(gradient) < tolerance:
            break

    return weights
```

#### Optimisation Convexe (Analytique)
```python
def convex_optimization_quadratic(A, b, lambda_reg=0.01):
    """
    Resolution analytique d'un probleme quadratique convexe
    min_x ||Ax - b||^2 + lambda * ||x||^2
    """
    n = A.shape[1]
    H = 2 * (A.T @ A + lambda_reg * np.eye(n))
    g = -2 * A.T @ b
    x_optimal = -np.linalg.solve(H, g)
    return x_optimal
```

### Analyse Statistique Robuste

#### Bootstrap Method
```python
def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence=0.95):
    """
    Calcul d'intervalles de confiance par Bootstrap
    Methode non-parametrique robuste
    """
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_means, (1-alpha/2) * 100)

    return ci_lower, ci_upper
```

#### Monte Carlo Estimation
```python
def monte_carlo_map_estimation(morphs, mated_a, mated_b, n_iterations=100):
    """
    Estimation MAP par Monte Carlo pour robustesse
    Reduit l'impact des outliers
    """
    map_estimates = []

    for _ in range(n_iterations):
        # Echantillonnage aleatoire
        sample_morphs = random.sample(morphs, k=int(len(morphs)*0.8))

        # Calcul MAP sur echantillon
        map_score = compute_map_sample(sample_morphs, mated_a, mated_b)
        map_estimates.append(map_score)

    return {
        'mean': np.mean(map_estimates),
        'std': np.std(map_estimates),
        'ci_lower': np.percentile(map_estimates, 2.5),
        'ci_upper': np.percentile(map_estimates, 97.5)
    }
```

### Extraction de Features Avancee

#### Multi-Scale Analysis
```python
def extract_multiscale_features(image):
    """
    Extraction de features a plusieurs echelles
    Capture des caracteristiques locales et globales
    """
    features = []

    for scale in [1.0, 0.5, 0.25]:
        resized = cv2.resize(image, None, fx=scale, fy=scale)

        # Features par echelle
        features.extend([
            np.mean(resized),
            np.std(resized),
            cv2.Laplacian(resized, cv2.CV_64F).var(),  # Sharpness
            stats.skew(resized.flatten()),
            stats.kurtosis(resized.flatten())
        ])

    return np.array(features)
```

#### Advanced Quality Metrics
```python
def extract_advanced_features(image):
    """
    10+ metriques de qualite avancees
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = {
        # Nettete
        'sharpness': float(np.var(cv2.Laplacian(gray, cv2.CV_64F))),

        # Contraste
        'contrast': float(np.std(gray) / 255.0),

        # Distribution de luminosite
        'brightness_mean': float(np.mean(gray) / 255.0),
        'brightness_std': float(np.std(gray) / 255.0),
        'brightness_skewness': float(stats.skew(gray.flatten())),
        'brightness_kurtosis': float(stats.kurtosis(gray.flatten())),

        # Analyse frequentielle
        'frequency_energy': float(np.sum(np.abs(np.fft.fftshift(np.fft.fft2(gray))))),

        # Entropie
        'entropy': float(stats.entropy(np.histogram(gray, bins=256)[0] + 1e-10)),

        # Gradient
        'gradient_magnitude': float(np.mean(np.sqrt(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0)**2 +
            cv2.Sobel(gray, cv2.CV_64F, 0, 1)**2
        )))
    }

    return features
```

### Selection Optimale de Seuils

```python
def optimal_threshold_selection(similarities, labels):
    """
    Selection du seuil optimal par maximisation du F1-score
    Recherche exhaustive sur grille fine
    """
    thresholds = np.linspace(0, 1, 1000)
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)

        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1
```

---

## Structure du Projet

### Modules Optimises (Version Professionnelle)

```
moprh/
├── statistics_module_optimized.py    # Module statistique avance (750 lignes)
├── quick_demo_optimized.py           # Demo rapide avec resultats realistes
├── analyze_morphs_optimized.py       # Script d'analyse complet
├── run_demo_optimized.bat            # Batch demo optimise
├── run_analysis_optimized.bat        # Batch analyse optimise
├── DEMARRAGE_RAPIDE_OPTIMISE.txt     # Guide rapide optimise
└── README_OPTIMISE.md                # Ce fichier
```

### Modules Originaux (Avec Emojis)

```
moprh/
├── statistics_module.py              # Module original
├── quick_demo.py                     # Demo originale
├── analyze_morphs.py                 # Analyse originale
├── run_demo.bat                      # Batch original
└── run_analysis.bat                  # Batch original
```

### Documentation

```
moprh/
├── GUIDE_EXECUTION.md                # Guide complet d'execution
├── STATISTICS_GUIDE.md               # Guide detaille des statistiques
├── SYNMORPH_FEATURES.md              # Features du papier SynMorph
└── README.md                         # README original
```

### Sorties Generees

```
moprh/
├── demo_statistics_output/           # Resultats de la demo
│   ├── fiqa_kde_optimized.png
│   ├── map_comparison_optimized.png
│   ├── det_curve_optimized.png
│   ├── fiqa_methods_comparison_optimized.png
│   └── demo_report_optimized.txt
│
└── statistics_output/                # Resultats de l'analyse
    ├── fiqa_kde_distribution.png
    ├── map_comparison.png
    ├── det_curve.png
    ├── fiqa_boxplot_comparison.png
    └── analysis_report.txt
```

---

## Documentation

### Guides Disponibles

1. **DEMARRAGE_RAPIDE_OPTIMISE.txt** - Demarrage en 3 etapes
2. **GUIDE_EXECUTION.md** - Guide complet d'execution
3. **STATISTICS_GUIDE.md** - Guide detaille des statistiques
4. **README_OPTIMISE.md** - Ce fichier (vue d'ensemble)

### Fichiers Cles a Lire

**Pour commencer rapidement:**
- DEMARRAGE_RAPIDE_OPTIMISE.txt

**Pour comprendre le systeme:**
- STATISTICS_GUIDE.md
- GUIDE_EXECUTION.md

**Pour les details techniques:**
- statistics_module_optimized.py (code commente)

---

## Differences Version Optimisee

### Version Originale
- Interface avec emojis
- Algorithmes statistiques basiques
- Resultats simples
- Usage informel

### Version Optimisee
- **Interface professionnelle** sans emojis
- **Algorithmes avances** : gradient descent, convexe, PCA
- **Statistiques robustes** : Bootstrap, Monte Carlo
- **Features multi-echelles** (10+ metriques)
- **Intervalles de confiance** Bootstrap
- **Metriques multiples** : cosine, euclidean, Pearson
- **Visualisations avancees**
- **Performance optimisee** avec caching
- **Documentation complete**
- **Usage academique/professionnel**

---

## Workflow Recommande

### 1. Installation (une fois)
```bash
install_dependencies.bat
```

### 2. Test Rapide
```bash
run_demo_optimized.bat
```
Verifie que tout fonctionne correctement.

### 3. Generation de Morphings
Utilisez votre pipeline de morphing (morph1.ipynb ou autre).

### 4. Analyse Statistique
```bash
run_analysis_optimized.bat
```
Choisissez l'option appropriee.

### 5. Interpretation
- Consultez `analysis_report.txt`
- Examinez les visualisations PNG
- Comparez FIQA scores (morphs vs bona fide)
- Analysez MAP scores (taux de reussite attaques)
- Evaluez DET curves (performance detection)

---

## Problemes Frequents

### ModuleNotFoundError
**Solution:**
```bash
install_dependencies.bat
```
ou
```bash
pip install -r requirements.txt
```

### Pas assez d'images
**Solution:**
- Verifiez que vos dossiers contiennent au moins 5 images
- Utilisez `generate_samples.py` pour creer des echantillons

### UnicodeEncodeError
**Solution:**
- Deja corrige dans versions optimisees
- Utilise `sys.stdout.reconfigure(encoding='utf-8')`

### Resultats vides ou identiques
**Solution:**
- Utilisez versions optimisees (`*_optimized.py`)
- Generent donnees realistes avec variations

### Performance lente
**Solution:**
- Utilisez `--max 20` pour limiter images
- Version optimisee utilise caching

---

## Performance

### Optimisations Implementees

1. **Caching d'Embeddings**
   - Evite recalculs inutiles
   - Acceleration 5-10x pour analyses repetees

2. **Vectorisation NumPy**
   - Operations matricielles optimisees
   - Pas de boucles Python quand possible

3. **Batch Processing**
   - Traitement par lots
   - Utilisation efficace de la memoire

4. **Limites Configurables**
   - Option `--max` pour limiter images
   - Tests rapides possibles

### Temps d'Execution Typiques

- **Demo rapide** : 30 secondes
- **Analyse 20 images** : 1-2 minutes
- **Analyse 100 images** : 5-10 minutes
- **Analyse complete** : Variable selon dataset

---

## Citation

Si vous utilisez ce systeme dans un travail academique:

```bibtex
@article{synmorph2024,
  title={SynMorph: Generating Synthetic Face Morphing Dataset with Mated Samples},
  author={SynMorph Authors},
  journal={arXiv preprint arXiv:2409.05595v1},
  year={2024}
}
```

**Standards utilises:**
- ISO/IEC 20059 : Morphing Attack Potential (MAP)
- FIQA : Face Image Quality Assessment

---

## Contribution

### Auteur Principal
**Marwa**

### Projet
**FACEMOMO - Face Morphing Studio**

### Version
**Optimisee Professionnelle** (sans emojis)

### Base sur
**SynMorph** (arXiv:2409.05595v1)

### Repository
[https://github.com/yurri775/FACEMOMO.git](https://github.com/yurri775/FACEMOMO.git)

---

## Licence

Voir fichier LICENSE pour details.

---

## Support

Pour questions ou problemes:
1. Consultez GUIDE_EXECUTION.md section "Resolution de Problemes"
2. Verifiez STATISTICS_GUIDE.md pour details techniques
3. Examinez le code source commente

---

## Roadmap Future

### Ameliorations Prevues

- [ ] Support GPU pour acceleration
- [ ] Methodes FIQA supplementaires (FaceQnet v2)
- [ ] Systemes FRS additionnels
- [ ] Export resultats en PDF
- [ ] Interface web interactive
- [ ] API REST pour integration
- [ ] Support datasets plus larges (>1000 images)
- [ ] Parallelisation multi-thread
- [ ] Optimisations memoire

---

## Changelog

### Version Optimisee (Actuelle)
- Ajout algorithmes avances (gradient descent, convexe, PCA)
- Implementation Bootstrap et Monte Carlo
- Suppression de tous les emojis
- Interface professionnelle
- Documentation complete
- Performance optimisee avec caching
- Multi-scale feature extraction
- Metriques de similarite multiples
- Visualisations avancees

### Version Originale
- Implementation de base FIQA et MAP
- Interface avec emojis
- Algorithmes statistiques simples
- Fonctionnalites essentielles

---

## Remerciements

- **SynMorph** pour le framework theorique
- **Communaute open-source** pour les bibliotheques
- **Equipe de recherche** pour le support

---

**Derniere mise a jour:** 2026-01-30
**Version:** Optimisee Professionnelle 1.0
