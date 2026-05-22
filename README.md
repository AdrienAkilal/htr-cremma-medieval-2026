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
   Fine-tuning + inférence → transcriptions ligne par ligne
       │
       ▼
5. Agrégation           src/aggregation/aggregate.py
   JSON data contract + flags needs_review + PAGE XML
       │
       ▼
6. Évaluation           src/evaluation/metrics.py
   CER · WER · IoU · bootstrap IC · McNemar
       │
       ▼
dataset_nlp/output.json  +  segmentations/*.page.xml
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

# 4. Tester le pipeline sur une image
python scripts/run_pipeline.py --image data/raw/Roman_de_la_Rose_BnF_fr25526.jpeg

# 5. Évaluer sur la validation
python scripts/evaluate.py --split val --model models/kraken_cremma_v1/best.mlmodel

# 6. Évaluation finale (une seule fois !)
python scripts/evaluate.py --split test --model models/kraken_cremma_v1/best.mlmodel

# 7. Lancer les tests
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
│   │   └── alto_parser.py              parser ALTO XML v4 ← NOUVEAU
│   ├── htr/kraken_htr.py               fine-tuning + inférence Kraken
│   ├── aggregation/aggregate.py        data contract JSON + PAGE XML
│   └── evaluation/metrics.py           CER · WER · IoU · bootstrap · McNemar
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_alto_parser.py             ← NOUVEAU
│   ├── test_metrics.py
│   └── test_aggregation.py
│
├── scripts/
│   ├── download_corpus.py      clone CREMMA-MSS-17 + parse ALTO
│   ├── make_splits.py          split par document + scellement SHA-256
│   ├── run_pipeline.py         pipeline complet sur une image
│   ├── train_kraken.py         fine-tuning Kraken
│   └── evaluate.py             évaluation CER/WER/IoU
│
├── experiments/journal.jsonl   log de toutes les expériences
├── dataset_nlp/                livrable JSON pour le module NLP
└── segmentations/              fichiers PAGE XML par folio
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
