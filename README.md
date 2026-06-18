# htr-cremma-medieval-2026

Pipeline HTR (Handwritten Text Recognition) pour manuscrits français du XVIIe siècle.
Corpus : **CREMMA-MSS-17** (HTR-United) — Master Data/IA, Module Vision par ordinateur, HETIC 2026.

**Équipe :** _(à compléter : noms des membres)_
**Responsable technique :** _(à compléter)_
**Responsable documentation :** _(à compléter)_

---

## Pipeline

```
Image brute (JPEG/TIFF)
       │
       ▼
1. Prétraitement        src/preprocessing/preprocess.py
   deskew · CLAHE · binarisation Sauvola
       │
       ▼
2. Segmentation layout  src/segmentation/yologen_obb.py
   YOLO-gen 11x-OBB → régions Text / Miniature / Initial / Decoration
       │
       ▼
3. Segmentation lignes  src/segmentation/kraken_segment.py
   Kraken BLLA → lignes de base + polygones
       │
       ▼
4. HTR Kraken           src/htr/kraken_htr.py
   Fine-tuning + inférence → transcriptions ligne par ligne + needs_review
       │
       ▼
5. Agrégation           src/aggregation/aggregate.py
   JSON data contract + flags needs_review + PAGE XML transcription CV
       │
       ▼
6. Normalisation NLP    src/nlp/normalize.py
   Règles procédurales : ⁊→et · (mot)→mot · [mot?]→mot · ꝑ→per/par
   Détection de langue · Transcription finale normalisée
       │
       ▼
output/{folio}/
  _segmentation.page.xml        (polygones/baselines Kraken)
  _transcription_cv.page.xml    (PAGE XML avec texte transcrit)
  _data_contract.json           (transcription CV complète)
  _nlp_normalized.json          (transcription normalisée + corrections)
  _transcription.txt            (texte brut lisible)
  _final_transcription.txt      (texte final normalisé)
```

---

## Installation

```bash
git clone https://github.com/VOTRE_ORG/htr-cremma-medieval-2026.git
cd htr-cremma-medieval-2026

python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Utilisation

### Lancer une transcription sur une image

```bash
# Détection automatique du modèle dans models/
python scripts/run_pipeline.py --image chemin/vers/photo.jpeg

# Avec un modèle spécifique
python scripts/run_pipeline.py \
    --image data/raw/CREMMA-MSS-17/data/recueil-lettres-pieces/f279.jpeg \
    --model models/kraken_cremma_v1/best.mlmodel

# Sans la segmentation layout YOLO (plus rapide)
python scripts/run_pipeline.py \
    --image data/raw/CREMMA-MSS-17/data/recueil-lettres-pieces/f279.jpeg \
    --skip-layout
```

Les sorties sont automatiquement générées dans `output/{nom_image}/`.

### Sorties dans `output/{folio}/`

| Fichier | Contenu |
|---------|---------|
| `{folio}_preprocessed.jpg` | Image de travail binarisée et nettoyée |
| `{folio}_segmentation.page.xml` | PAGE XML de segmentation (polygones + baselines) |
| `{folio}_transcription_cv.page.xml` | PAGE XML avec le texte transcrit par Kraken |
| `{folio}_data_contract.json` | Transcription CV complète avec `confidence` et `needs_review` par ligne |
| `{folio}_transcription.txt` | Transcription brute lisible avec flag `[REVIEW]` |
| `{folio}_nlp_normalized.json` | Transcription normalisée avec toutes les corrections tracées |
| `{folio}_final_transcription.txt` | Transcription finale normalisée lisible |

### Format de `_nlp_normalized.json`

```json
{
  "page_id": "f279",
  "lines": [
    {
      "line_id": "l_0001",
      "transcription_cv":         "du saint esperit ⁊ par le (com)mandement",
      "transcription_normalisee": "du saint esperit et par le commandement",
      "corrections_appliquees": [
        {"position": 18, "avant": "⁊",     "apres": "et",  "type": "normalisation_tironien"},
        {"position": 31, "avant": "(com)", "apres": "com", "type": "validation_abreviation"}
      ],
      "langue_detectee": ["fr"],
      "needs_review": false,
      "confidence": 0.92
    }
  ],
  "stats": {
    "n_lines": 42,
    "n_processed": 38,
    "n_skipped_review": 4,
    "n_corrections": 17
  }
}
```

### Règles de normalisation NLP appliquées

| # | Marqueur / règle | Résultat | Description |
|---|---|---|---|
| 1 | encodage Unicode | NFC | Fusion des caractères décomposés |
| 2 | `u` initial de mot | `v` | Allographe u/v (ex. `uous` → `vous`) |
| 3 | `i` initial + voyelle | `j` | Allographe i/j (ex. `ie` → `je`) |
| 4 | `⁊` | `et` | Note tironienne |
| 5 | `ꝑ` | `per` ou `par` | P barré (heuristique sur le caractère suivant) |
| 6 | `(mot)` | `mot` | Développement d'abréviation validé |
| 7 | `[mot?]` | `mot` | Lecture incertaine confirmée |
| 8 | `<R>texte</R>` | `texte` | Balise de rubrique supprimée |
| 9 | `\|\|` | espace | Séparateur de colonne |
| — | `[?]` `[...]` `[†]` `[abr]` | conservés | Irréductibles sans contexte |

Les lignes avec `needs_review=true` (confiance < 0.80) ne sont **pas normalisées**.

#### Impact chiffré — évaluation relative (f12 + f279, 33 lignes processables)

| Règle                 | Corrections | Distance relative |
|-----------------------|-------------|-------------------|
| NFC                   | 2           | 0.39 %            |
| u/v (u initial → v)   | 11          | 1.07 %            |
| i/j (i initial → j)   | 6           | 0.58 %            |
| Tironien → et         | 1           | 0.19 %            |
| P barré → per/par     | 0           | 0.00 %            |
| Abréviations (mot)    | 0           | 0.00 %            |
| Lectures [mot?]       | 0           | 0.00 %            |
| Rubriques \<R>        | 0           | 0.00 %            |
| Séparateur \|\|       | 0           | 0.00 %            |
| **TOTAL**             | **20**      | **2.24 %**        |

> La distance relative mesure `Levenshtein(brut, normalisé) / len(brut)`.
> Sans vérité terrain sur les manuscrits BnF, c'est l'évaluation de référence
> (cf. consignes NLP). Relancer avec `python scripts/compute_cer.py --all`.

### Script de transcription simple (évaluation avec vérité terrain)

```bash
# Sans vérité terrain
python scripts/transcribe.py --image data/raw/.../f279.jpeg

# Avec vérité terrain ALTO XML (calcule CER / ACC)
python scripts/transcribe.py \
    --image data/raw/CREMMA-MSS-17/data/recueil-lettres-pieces/f279.jpeg \
    --xml   data/raw/CREMMA-MSS-17/data/recueil-lettres-pieces/f279.xml \
    --model models/kraken_cremma_v1/model_10.mlmodel
```

---

## Reproduire les résultats

```bash
# 1. Télécharger et parser le corpus CREMMA-MSS-17
python scripts/download_corpus.py
#    → clone https://github.com/HTR-United/CREMMA-MSS-17
#    → parse les ALTO XML → data/raw/cremma_mss17_manifest.json

# 2. Constituer les splits train/val/test (split par document, seed=42)
python scripts/make_splits.py
#    → data/splits/train.json · val.json · test.json
#    → data/splits/test_sha256.txt  ← hash du test scellé

# 3. Fine-tuner Kraken sur le corpus
python scripts/train_kraken.py

# 4. Lancer le pipeline complet sur une image
python scripts/run_pipeline.py --image data/raw/CREMMA-MSS-17/data/recueil-lettres-pieces/f279.jpeg

# 5. Évaluer sur la validation
python scripts/evaluate.py --split val --model models/kraken_cremma_v1/best.mlmodel

# 6. Évaluation finale (une seule fois !)
python scripts/evaluate.py --split test --model models/kraken_cremma_v1/best.mlmodel

# 7. Lancer les tests unitaires
pytest tests/
```

---

## Structure du dépôt

```
htr-cremma-medieval-2026/
├── README.md
├── requirements.txt
├── pyproject.toml
├── CONVENTIONS_TRANSCRIPTION.md
├── DATA_SOURCES.md
├── MODEL_CARD.md
│
├── data/
│   ├── raw/
│   │   ├── CREMMA-MSS-17/          ← repo cloné (git clone)
│   │   └── cremma_mss17_manifest.json
│   └── splits/
│       ├── train.json
│       ├── val.json
│       ├── test.json               ← scellé, ne jamais regarder avant le rendu
│       └── test_sha256.txt
│
├── src/
│   ├── preprocessing/preprocess.py     deskew · CLAHE · Sauvola
│   ├── segmentation/
│   │   ├── yologen_obb.py              segmentation layout (YOLO-gen)
│   │   ├── kraken_segment.py           segmentation lignes (Kraken BLLA)
│   │   └── alto_parser.py              parser ALTO XML v4
│   ├── htr/kraken_htr.py               fine-tuning + inférence Kraken
│   ├── aggregation/aggregate.py        data contract JSON + PAGE XML
│   ├── nlp/normalize.py                normalisation NLP procédurale    ← NOUVEAU
│   └── evaluation/metrics.py           CER · WER · IoU · bootstrap · McNemar
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_alto_parser.py
│   ├── test_metrics.py
│   ├── test_aggregation.py
│   └── test_nlp.py                     ← NOUVEAU
│
├── scripts/
│   ├── download_corpus.py      clone CREMMA-MSS-17 + parse ALTO
│   ├── make_splits.py          split par document + scellement SHA-256
│   ├── run_pipeline.py         pipeline complet (étapes 1→6)
│   ├── transcribe.py           transcription simple avec métriques CER
│   ├── train_kraken.py         fine-tuning Kraken
│   └── evaluate.py             évaluation CER/WER/IoU
│
├── output/                     sorties par folio (créé à l'exécution)
│   └── {folio}/
│       ├── {folio}_preprocessed.jpg
│       ├── {folio}_segmentation.page.xml
│       ├── {folio}_transcription_cv.page.xml
│       ├── {folio}_data_contract.json
│       ├── {folio}_transcription.txt
│       ├── {folio}_nlp_normalized.json
│       └── {folio}_final_transcription.txt
│
├── experiments/journal.jsonl   log de toutes les expériences
├── dataset_nlp/                livrable JSON pour le module NLP
└── segmentations/              fichiers PAGE XML par folio (eScriptorium)
```

---

## Organisation de l'équipe

| Rôle | Membre |
|------|--------|
| Responsable technique | _(à compléter)_ |
| Responsable documentation | _(à compléter)_ |
| Responsable expérimentation | _(à compléter)_ |
| Responsable données | _(à compléter)_ |

---

## Résultats (test scellé)

| Métrique | Valeur | Seuil validation | Seuil excellence |
|----------|--------|-----------------|-----------------|
| CER global | _en cours_ | < 15 % | < 8 % |
| WER global | _en cours_ | < 25 % | < 15 % |
| IoU segmentation | _en cours_ | > 0.75 | > 0.85 |
| Taux needs_review | _en cours_ | < 30 % | < 20 % |

**SHA-256 test set :** _(généré par `make_splits.py` — à copier ici)_
