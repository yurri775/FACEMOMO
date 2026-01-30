# 🚀 Guide d'Exécution - Système de Statistiques

## ⚡ Exécution Rapide (Recommandé pour commencer)

### Option 1 : Démonstration Complète

La façon la plus simple de tester le système:

```bash
# Ouvrez un terminal dans le dossier moprh/ et exécutez:
python quick_demo.py
```

**Ce que ça fait:**
- ✅ Crée des données de démonstration
- ✅ Analyse FIQA avec 3 méthodes
- ✅ Calcule le MAP sur 4 FRS
- ✅ Génère tous les graphiques
- ✅ Crée un rapport complet
- 📁 Résultats dans `demo_statistics_output/`

**Durée:** ~30 secondes

---

### Option 2 : Test du Module de Statistiques

Pour voir les fonctionnalités du module:

```bash
python statistics_module.py
```

**Ce que ça fait:**
- ✅ Démo des classes FIQAAnalyzer, MAPAnalyzer, StatisticsVisualizer
- ✅ Exemples d'utilisation de chaque fonction
- 📁 Résultats dans `statistics_output/`

**Durée:** ~20 secondes

---

## 📊 Analyse de Vos Données Réelles

### Étape 1 : Avoir des Images

Vous devez avoir deux dossiers:
1. **Images morphées** (morphing results)
2. **Images originales** (bona fide)

**Exemple avec les échantillons de démonstration:**

```bash
# Analyser les échantillons existants
python analyze_morphs.py --morph sample_data/after_morph --bona-fide sample_data/before_morph
```

**Exemple avec vos propres morphings:**

```bash
# D'abord générer des morphings si vous ne l'avez pas fait
# Ouvrez morph1.ipynb dans Jupyter et exécutez toutes les cellules
# OU utilisez:
python generate_samples.py

# Puis analyser vos résultats
python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph
```

**Durée:** Variable selon le nombre d'images (1-5 minutes pour ~50 images)

---

## 🎯 Guide Pas à Pas Complet

### 1️⃣ Installation des Dépendances

**Windows:**
```bash
# Double-cliquez sur:
install_dependencies.bat

# OU dans un terminal:
pip install numpy opencv-python dlib matplotlib scikit-learn pillow imageio tqdm seaborn scipy
```

**Linux/Mac:**
```bash
pip install numpy opencv-python dlib matplotlib scikit-learn pillow imageio tqdm seaborn scipy
```

---

### 2️⃣ Première Démonstration

```bash
# Test rapide du système
python quick_demo.py
```

**Résultats attendus:**
```
✓ Création de 5 morphs et 5 images originales
✓ Analyse FIQA (3 méthodes)
✓ Analyse MAP (4 FRS)
✓ 4 graphiques générés
✓ 1 rapport texte
```

**Vérifier les résultats:**
```bash
# Ouvrez le dossier:
demo_statistics_output/

# Vous devriez voir:
├── demo_fiqa_kde.png
├── demo_map_comparison.png
├── demo_det_curve.png
└── demo_report.txt
```

---

### 3️⃣ Générer des Morphings (Si Nécessaire)

Si vous n'avez pas encore de morphings:

**Option A : Script Python**
```bash
python generate_samples.py
```
Génère 5 échantillons dans `sample_data/`

**Option B : Jupyter Notebook (Plus de contrôle)**
```bash
# Installer Jupyter si nécessaire
pip install jupyter

# Lancer Jupyter
jupyter notebook

# Dans le navigateur, ouvrir morph1.ipynb
# Exécuter toutes les cellules (Cell > Run All)
```

---

### 4️⃣ Analyser Vos Morphings

Une fois que vous avez des morphings générés:

```bash
# Analyse complète
python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph

# OU avec les échantillons de démo
python analyze_morphs.py --morph sample_data/after_morph --bona-fide sample_data/before_morph

# Pour un test rapide (limite à 20 images)
python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph --max 20
```

**Paramètres disponibles:**
- `--morph <dossier>` : Dossier des images morphées
- `--bona-fide <dossier>` : Dossier des images originales
- `--output <dossier>` : Dossier de sortie (défaut: statistics_output)
- `--max <nombre>` : Limite le nombre d'images

---

### 5️⃣ Consulter les Résultats

```bash
# Les résultats sont dans:
statistics_output/

# Ouvrir les images:
# - fiqa_kde_simple.png
# - fiqa_kde_facequnet.png
# - fiqa_kde_serfiq.png
# - map_comparison.png
# - det_curve_fiqa.png
# - fiqa_methods_comparison.png

# Lire le rapport:
# - analysis_report.txt
```

**Pour ouvrir rapidement:**

Windows:
```bash
explorer statistics_output
```

Linux/Mac:
```bash
open statistics_output    # Mac
xdg-open statistics_output  # Linux
```

---

## 📝 Utilisation Programmatique (Python)

Si vous voulez intégrer dans votre propre code:

```python
from statistics_module import FIQAAnalyzer, MAPAnalyzer, StatisticsVisualizer
import cv2
from pathlib import Path

# 1. Charger vos images
morph_dir = Path("morphing_results")
morph_images = [cv2.imread(str(f)) for f in morph_dir.glob("*.png")]

bona_fide_dir = Path("sample_data/before_morph")
bona_fide_images = [cv2.imread(str(f)) for f in bona_fide_dir.glob("*.png")]

# 2. Préparer les données
all_images = morph_images + bona_fide_images
labels = ['morph'] * len(morph_images) + ['bona_fide'] * len(bona_fide_images)

# 3. Analyse FIQA
fiqa = FIQAAnalyzer()
stats = fiqa.analyze_dataset(all_images, labels, method='simple')

print(f"Qualité moyenne morphs: {stats['morph']['mean']:.3f}")
print(f"Qualité moyenne originaux: {stats['bona_fide']['mean']:.3f}")
print(f"KL-Divergence: {stats['kl_divergence']:.4f}")

# 4. Analyse MAP
map_analyzer = MAPAnalyzer()
mid = len(bona_fide_images) // 2
map_results = map_analyzer.compute_map(
    morph_images[:10],
    bona_fide_images[:mid],
    bona_fide_images[mid:],
    threshold=0.6
)

for model, results in map_results.items():
    print(f"{model}: MAP = {results['map_score']:.3f}")

# 5. Visualisations
viz = StatisticsVisualizer(output_dir="my_analysis")
viz.plot_kde_comparison(stats)
viz.plot_map_comparison(map_results)
viz.generate_summary_report(stats, map_results)
```

---

## 🔧 Résolution de Problèmes

### Problème 1 : "ModuleNotFoundError"
```
ModuleNotFoundError: No module named 'seaborn'
```

**Solution:**
```bash
pip install seaborn scipy
```

---

### Problème 2 : "Pas assez d'images"
```
❌ ERREUR: Pas assez d'images chargées pour l'analyse!
```

**Solution:**
- Vérifiez que les dossiers existent et contiennent des images
- Vérifiez les extensions (.png, .jpg)
- Générez d'abord des échantillons avec `python generate_samples.py`

---

### Problème 3 : Encodage Unicode (Windows)
```
UnicodeEncodeError: 'charmap' codec can't encode characters
```

**Solution:**
Le code gère déjà ceci, mais si le problème persiste:
```bash
# Définir l'encodage UTF-8
set PYTHONIOENCODING=utf-8
python analyze_morphs.py
```

---

### Problème 4 : Mémoire insuffisante
```
MemoryError
```

**Solution:**
```bash
# Limiter le nombre d'images
python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph --max 20
```

---

## 📊 Exemples de Commandes Complètes

### Scénario 1 : Test Initial
```bash
# 1. Installer les dépendances
pip install seaborn scipy matplotlib

# 2. Test rapide
python quick_demo.py

# 3. Vérifier les résultats
explorer demo_statistics_output  # Windows
```

---

### Scénario 2 : Analyse Complète
```bash
# 1. Générer des échantillons (si besoin)
python generate_samples.py

# 2. Générer des morphings complets
# Ouvrir morph1.ipynb dans Jupyter et exécuter

# 3. Analyser
python analyze_morphs.py --morph morphing_results --bona-fide sample_data/before_morph

# 4. Consulter
explorer statistics_output
```

---

### Scénario 3 : Test Rapide sur Échantillons
```bash
# Analyse directe des échantillons de démo
python analyze_morphs.py

# Les paramètres par défaut sont:
# --morph sample_data/after_morph
# --bona-fide sample_data/before_morph
```

---

## ⏱️ Temps d'Exécution Estimés

| Action | Nombre d'images | Temps estimé |
|--------|----------------|--------------|
| `quick_demo.py` | 10 (générées) | ~30 secondes |
| `statistics_module.py` | 40 (générées) | ~20 secondes |
| `analyze_morphs.py` (échantillons) | 10 réelles | ~1 minute |
| `analyze_morphs.py` (50 images) | 50 réelles | ~3 minutes |
| `analyze_morphs.py` (--max 20) | 20 réelles | ~1.5 minutes |

---

## 📚 Fichiers Importants

| Fichier | Description | Commande |
|---------|-------------|----------|
| `quick_demo.py` | Démo rapide avec données simulées | `python quick_demo.py` |
| `statistics_module.py` | Module de base (peut être testé) | `python statistics_module.py` |
| `analyze_morphs.py` | Script d'analyse complet | `python analyze_morphs.py` |
| `generate_samples.py` | Génère des échantillons | `python generate_samples.py` |
| `morph1.ipynb` | Notebook Jupyter principal | Ouvrir avec Jupyter |

---

## 🎓 Pour Votre Professeur

Séquence de démonstration recommandée:

```bash
# 1. Montrer la démo rapide
python quick_demo.py

# 2. Montrer l'analyse sur les échantillons réels
python analyze_morphs.py --morph sample_data/after_morph --bona-fide sample_data/before_morph

# 3. Ouvrir les résultats
explorer statistics_output

# 4. Montrer le rapport texte
type statistics_output\analysis_report.txt  # Windows
cat statistics_output/analysis_report.txt   # Linux/Mac
```

---

## ✅ Checklist de Vérification

Avant de présenter:

- [ ] `pip install seaborn scipy` exécuté
- [ ] `python quick_demo.py` fonctionne
- [ ] Dossier `demo_statistics_output/` créé avec 4 fichiers
- [ ] `python analyze_morphs.py` fonctionne
- [ ] Dossier `statistics_output/` créé avec 7 fichiers
- [ ] Les images PNG s'ouvrent correctement
- [ ] Le fichier `analysis_report.txt` est lisible

---

**Besoin d'aide?** Consultez:
- [STATISTICS_GUIDE.md](STATISTICS_GUIDE.md) - Guide détaillé des statistiques
- [SYNMORPH_FEATURES.md](SYNMORPH_FEATURES.md) - Fonctionnalités du papier
- [README.md](README.md) - Documentation générale
