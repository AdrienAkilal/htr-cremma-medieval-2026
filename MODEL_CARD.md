# Model Card — Kraken CREMMA Medieval 2026

## Présentation

Modèle Kraken fine-tuné sur le corpus CREMMA Medieval pour la reconnaissance automatique d'écriture manuscrite (HTR) en ancien et moyen français (XIIIe–XVe siècle).

## Performances

| Métrique | Val set | Test set |
|----------|---------|----------|
| CER | _à compléter_ | _à compléter_ |
| WER | _à compléter_ | _à compléter_ |
| IC 95% CER | _à compléter_ | _à compléter_ |

## Données d'entraînement

- **Corpus** : CATMuS Medieval + CREMMA Medieval
- **Volume** : _à compléter_ lignes
- **Période** : XIIIe–XVe siècle
- **Langue** : ancien et moyen français

## Limitations

- Performances dégradées sur les écritures cursives tardives (XVIe s.)
- Abréviations rares sous-représentées dans le corpus
- Textes bilingues (latin/français) non gérés
- _À compléter après expériences_

## Utilisation

```python
from kraken.lib import models
from kraken import rpred

model = models.load_any("models/kraken_cremma_v1/best.mlmodel")
```

## Citation

```bibtex
@misc{htr-cremma-medieval-2026,
  title  = {HTR Pipeline for Medieval French Manuscripts},
  author = {Équipe MD5 HETIC 2026},
  year   = {2026},
  url    = {https://github.com/VOTRE_ORG/htr-cremma-medieval-2026}
}
```
