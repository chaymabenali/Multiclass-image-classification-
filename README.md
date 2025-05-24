# Multiclass-image-classification-
Ce projet met en œuvre un pipeline complet de vision par ordinateur et d'apprentissage automatique pour classifier des images d'instruments de musique en plusieurs catégories distinctes. Il couvre les étapes allant de la préparation des données à l'évaluation de modèles de classification,en passant par la segmentation d'objets et l'extraction de caractéristiques.

## Objectif du Projet
L'objectif principal de ce projet est de développer un système capable d'identifier et de classifier automatiquement différents types d'instruments de musique à partir d'images. Ce projet explore l'application de techniques de segmentation modernes et l'extraction de caractéristiques classiques pour alimenter des modèles de Machine Learning.

## Dataset
Le projet utilise un dataset d'images représentant plusieurs classes d'instruments de musique. Les classes incluses sont :
`accordion`, `banjo`, `drum`, `flute`, `guitar`, `harmonica`, `saxophone`, `sitar`, `tabla`, `violin`.

Le dataset initial est divisé en ensembles d'entraînement (80%) et de test (20%) pour chaque classe d'instrument.

## Pipeline

### 1. Préparation des Données
- Les images du dataset original sont décompressées.
- Les données sont divisées en ensembles d'entraînement et de test, en conservant la structure des classes.
- Des étapes de prétraitement basiques (redimensionnement, normalisation) sont appliquées.

### 2. Segmentation d'Objets avec YOLOv8
- Le modèle de segmentation pré-entraîné `yolov8n-seg.pt` (ou une variante) est utilisé pour détecter les objets dans chaque image.
- Les masques de tous les objets détectés dans une image sont combinés (par un OU logique) pour créer un masque binaire unique pour l'image. L'objectif est que ce masque capture l'instrument principal.
- Les masques générés sont sauvegardés pour une utilisation ultérieure.

### 3. Extraction de Caractéristiques
À partir des images originales et des masques binaires générés par YOLOv8, un ensemble de caractéristiques est extrait pour chaque contour principal détecté dans les masques :
- **Caractéristiques Géométriques :** Aire, périmètre, rapport d'aspect, circularité.
- **Caractéristiques Photométriques :** Intensité moyenne et écart-type des pixels dans la région masquée.
- **Caractéristiques de Texture (GLCM) :** Énergie, homogénéité, entropie de Shannon (calculées sur la matrice de co-occurrence des niveaux de gris).
Les caractéristiques extraites sont stockées dans des fichiers CSV (`instrument_train_features.csv`, `instrument_test_features.csv`).

### 4. Classification Machine Learning
- Les caractéristiques extraites sont utilisées pour entraîner et évaluer plusieurs modèles de classification :
    - Support Vector Machine (SVM)
    - k-Nearest Neighbors (k-NN)
    - Random Forest (Forêt Aléatoire)
    - Régression Logistique
    - Gaussian Naive Bayes
    - Arbre de Décision
- Les performances des modèles sont évaluées à l'aide de métriques telles que l'accuracy, la précision, le rappel, le F1-score (pondéré pour le multi-classe), les matrices de confusion, les courbes d'apprentissage et les courbes ROC (One-vs-Rest).

### 5. Analyse des Caractéristiques
- Une analyse exploratoire est menée sur les caractéristiques extraites de l'ensemble d'entraînement pour :
    - Visualiser la distribution de chaque caractéristique par classe.
    - Générer des matrices de corrélation.
    - Effectuer une Analyse en Composantes Principales (PCA) pour visualiser la séparabilité des classes dans un espace de dimension réduite.

## Technologies Utilisées
- **Langage :** Python 3.x
- **Bibliothèques principales :**
    - OpenCV (`cv2`): Traitement d'images, lecture/écriture, transformations.
    - Scikit-learn (`sklearn`): Modèles de Machine Learning, métriques, prétraitement, PCA.
    - Ultralytics YOLO (`ultralytics`): Segmentation d'objets.
    - NumPy: Calcul numérique.
    - Pandas: Manipulation de données (DataFrames pour les caractéristiques).
    - Matplotlib & Seaborn: Visualisation des données et des résultats.
    - Scikit-image (`skimage`): Pour les caractéristiques GLCM.
    - `tqdm`: Barres de progression.
- **Environnement :** Google Colaboratory (ou environnement Python local avec les dépendances installées).
