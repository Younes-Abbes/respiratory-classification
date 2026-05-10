# Projet TP GL4-RT4 — Classification des sons respiratoires (ICBHI 2017)

**Équipe :** Adem Saidi · Rami Gharbi · Adem Letaief Boukamcha
**Deadline :** 03 avril 2026
**Article de référence :** Işık et al., *« Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers »*, arXiv:2512.22564 ([dépôt officiel](https://github.com/Atakanisik/ICBHI-AST-SAM)).

---

## 1. Objectif

Améliorer la **sensibilité (rappel)** d'un modèle de classification automatique des sons respiratoires sur le dataset *ICBHI 2017 Challenge*, par rapport au modèle de référence (AST + SAM) qui atteint **Se = 68.31 %**, **Sp = 67.89 %**, **Score ICBHI = 68.10 %**.

Notre démarche :

1. **Reproduction** exacte du baseline du papier (config `configs/baseline.yaml`).
2. **Améliorations** ajoutées de manière incrémentale (config `configs/improved.yaml`) :
   - Class-Balanced Focal Loss
   - SpecAugment (masquage temps + fréquence)
   - Mixup au niveau spectrogramme
   - Augmentation au niveau forme d'onde (bruit, gain, décalage)
   - **Adaptive SAM (ASAM)** au lieu du SAM vanille
   - Scheduler cosinus avec warmup
   - Learning-rate différentiel (tête > backbone)
   - Test-Time Augmentation à l'évaluation
3. **Comparaison** systématique avec le papier de référence et étude d'ablation.

---

## 2. Structure du projet

```
ICBHI-AST-SAM-Improved/
├── README.md                         <- ce fichier
├── requirements.txt
├── configs/
│   ├── baseline.yaml                 <- reproduit le papier
│   └── improved.yaml                 <- nos améliorations
├── src/
│   ├── __init__.py
│   ├── augmentations.py              <- WaveformAugment, SpecAugment, Mixup
│   ├── dataset.py                    <- ASTDataset, WeightedRandomSampler
│   ├── losses.py                     <- FocalLoss, class-balanced alpha
│   ├── metrics.py                    <- Se / Sp / Score officiel ICBHI
│   ├── model.py                      <- Wrapper AST + tête de classification
│   ├── sam.py                        <- SAM et ASAM (Adaptive SAM)
│   └── utils.py                      <- Seed, config YAML, AverageMeter
├── preprocess.py                     <- Audio → .npz prêt à entraîner
├── train.py                          <- Boucle d'entraînement (SAM + augm.)
├── evaluate.py                       <- Eval + matrice de confusion + t-SNE + TTA
├── notebooks/
│   └── colab_quickstart.ipynb        <- Pipeline complet sur Colab gratuit
└── docs/
    └── Rapport_Projet_GL4_RT4.docx   <- Rapport final
```

---

## 3. Installation

### Local (avec GPU CUDA)

```bash
git clone <votre_repo>
cd ICBHI-AST-SAM-Improved
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab (recommandé — GPU gratuit)

Ouvrir `notebooks/colab_quickstart.ipynb` dans Colab, choisir un runtime GPU (T4 suffit largement), puis exécuter les cellules dans l'ordre. Le notebook :

1. clone ce dépôt,
2. télécharge automatiquement le dataset ICBHI,
3. exécute le préprocessing,
4. lance l'entraînement (~45 minutes pour 20 epochs sur T4),
5. évalue le modèle.

---

## 4. Préparation du dataset ICBHI 2017

Le dataset n'est **pas redistribuable** ; il faut le télécharger depuis le portail officiel :

```bash
mkdir -p data && cd data

# Base de 920 fichiers .wav + .txt
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
unzip ICBHI_final_database.zip

# Split officiel train/test
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt

cd ..
```

Arborescence attendue :

```
data/
├── ICBHI_final_database/
│   ├── 101_1b1_Al_sc_Meditron.wav
│   ├── 101_1b1_Al_sc_Meditron.txt
│   └── ... (920 fichiers .wav + 920 .txt)
└── ICBHI_challenge_train_test.txt
```

---

## 5. Préprocessing

```bash
python preprocess.py \
    --data_dir   ./data/ICBHI_final_database \
    --split_file ./data/ICBHI_challenge_train_test.txt \
    --output_path ./icbhi_ast_16k_8s.npz
```

Le script effectue :

1. Lecture du split officiel.
2. Pour chaque fichier `.wav` : chargement à **16 kHz**.
3. Découpe en cycles respiratoires d'après les annotations `.txt`.
4. **Cyclic padding** sur **8 secondes** (128 000 échantillons) — préserve le signal sans introduire de silences artificiels.
5. Étiquetage 4 classes : `0=Normal, 1=Crackle, 2=Wheeze, 3=Both`.
6. Sauvegarde en `.npz` (~3.5 Go).

---

## 6. Entraînement

### Reproduction du baseline (papier)

```bash
python train.py --config configs/baseline.yaml
```

### Notre version améliorée

```bash
python train.py --config configs/improved.yaml
```

Le script affiche, à chaque epoch, la sensibilité, la spécificité et le score ICBHI mesurés sur le test set officiel, et sauvegarde automatiquement le meilleur checkpoint (`checkpoints/<exp_name>_best.pth`).

### Override en ligne de commande

Tous les paramètres du YAML peuvent être overridés. Exemple :

```bash
python train.py --config configs/improved.yaml \
    --epochs 30 --batch_size 16 --rho 0.7 --use_asam
```

---

## 7. Évaluation

Avec **Test-Time Augmentation** activé et génération des graphiques (matrice de confusion + t-SNE) :

```bash
python evaluate.py \
    --model_path ./checkpoints/improved_ast_asam_best.pth \
    --output_dir ./results \
    --use_tta --tta_n 4 \
    --make_tsne
```

Le script affiche les métriques officielles puis enregistre :

- `results/confusion_matrix.png` : matrice 4 × 4.
- `results/tsne.png` : projection 2D des embeddings AST.

---

## 8. Améliorations proposées par rapport au papier

| # | Amélioration | Justification | Implémentation |
|---|---|---|---|
| 1 | **Class-Balanced Focal Loss** | La spécificité du papier (67.89 %) est la métrique faible. Focal Loss + alpha basé sur le « nombre effectif d'échantillons » (Cui et al. 2019) cible mieux les exemples difficiles que la simple CE + sampling. | `src/losses.py` |
| 2 | **SpecAugment** | Standard pour les modèles audio. Masque aléatoirement des bandes de fréquences et fenêtres temporelles : régularise sans nécessiter de données supplémentaires. | `src/augmentations.py` |
| 3 | **Mixup** | Combinaison convexe de spectrogrammes. Force le modèle à apprendre des frontières de décision plus lisses et combat le sur-apprentissage. | `src/augmentations.py` |
| 4 | **Augmentation forme d'onde** | Bruit gaussien à SNR contrôlé, décalage temporel, gain. Simule la variabilité réelle des conditions de capture. | `src/augmentations.py` |
| 5 | **Adaptive SAM (ASAM)** | Variante de SAM scale-invariante. Permet un `rho` plus grand (0.5 vs 0.05) et donne de meilleurs résultats sur petit dataset déséquilibré. | `src/sam.py` |
| 6 | **Cosine LR + warmup** | Le papier utilise un LR constant. Un warmup évite de déstabiliser les poids AudioSet pré-entraînés ; la décroissance cosinus améliore la convergence finale. | `train.py:build_scheduler` |
| 7 | **LR différentiel** | Tête neuve (×10) vs backbone pré-entraîné (×1). Empêche le backbone d'être écrasé par les gradients bruyants de la tête. | `model.py:get_param_groups` |
| 8 | **Test-Time Augmentation** | Moyenner les logits sur 4-5 augmentations légères du test stabilise les prédictions sans modifier l'entraînement. | `evaluate.py:predict_logits_tta` |

---

## 9. Résultats attendus et étude d'ablation

À compléter par votre équipe **après** chaque exécution d'entraînement. Modèle de tableau :

| Configuration | Se (%) | Sp (%) | Score (%) |
|---|---|---|---|
| Baseline AST (papier, ré-implémenté) | … | … | … |
| + WeightedRandomSampler (papier) | … | … | … |
| + SAM (papier — référence) | **68.31** | **67.89** | **68.10** |
| + ASAM | … | … | … |
| + ASAM + SpecAugment | … | … | … |
| + ASAM + SpecAugment + Mixup | … | … | … |
| + ASAM + SpecAugment + Mixup + CB-Focal | … | … | … |
| + tout + LR diff. + cosine warmup | … | … | … |
| + tout + TTA | … | … | … |

---

## 10. Reproductibilité

- Toutes les graines (Python, NumPy, torch, CUDA) sont fixées via `set_seed` (par défaut 42).
- Les versions exactes sont dans `requirements.txt`.
- En raison du non-déterminisme de certains kernels CUDA (notamment `cudnn` benchmark), des variations de ±0.5 % sont normales.

---

## 11. Livrables

- `Rapport_Projet_GL4_RT4.docx` (dans `docs/`) — rapport décrivant l'architecture, le préprocessing, l'entraînement, les améliorations et les résultats.
- Code source complet (ce dépôt).
- Checkpoints `.pth` à fournir séparément si demandés (trop volumineux pour le zip).

---

## Licence

Code distribué sous **MIT**, à des fins éducatives et de recherche.
Le dataset ICBHI 2017 reste sous sa licence d'origine — voir [https://bhichallenge.med.auth.gr/](https://bhichallenge.med.auth.gr/).
