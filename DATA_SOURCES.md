# Sources des données

## Corpus principal

| Source | Licence | Période | Description |
|--------|---------|---------|-------------|
| [CREMMA-MSS-17](https://github.com/HTR-United/CREMMA-MSS-17) | CC-BY 4.0 | XVIIe s. | GT HTR pour manuscrits du 17e siècle — 6 sous-corpus, ~130 fichiers ALTO |

### Sous-corpus CREMMA-MSS-17

| Sous-corpus | Type | Fichiers | Source |
|-------------|------|----------|--------|
| recueil-lettres-pieces | Lettres | 11 | Microfilm |
| dépêches-originales-adressées-à-la-cour-par-divers | Correspondance | 25 | Microfilm |
| lettres-de-bossuet | Lettres | 18 | Microfilm |
| correspondance-dom-bernard-de-montfaucon | Correspondance | 20 | Couleur |
| correspondance-dom-bernard-de-montfaucon-vol3 | Correspondance | 29 | Couleur |
| pensées-sur-la-religion-par-blaise-pascal | Littérature | — | Couleur |

## Format des données

Les fichiers sont en **ALTO XML v4** (standard Library of Congress), produits avec
eScriptorium et Kraken, suivant les normes de segmentation **SegmOnto**.

Structure dans le repo :
```
data/
  <nom-sous-corpus>/
    *.xml    ← ALTO XML (segmentation + transcription)
    *.jpg    ← images (ou lien Gallica)
```

## Images de test

| Image | Source | Cote | Licence |
|-------|--------|------|---------|
| Roman de la Rose | [Gallica BnF](https://gallica.bnf.fr) | Français 25526 | Domaine public |
| Protocole / registre de minutes | [Gallica BnF](https://gallica.bnf.fr) | NAF 1287 | Domaine public |

## Modèles pré-entraînés

| Modèle | Source | Licence | Usage |
|--------|--------|---------|-------|
| YOLO-gen 11x-OBB | [magistermilitum/YOLO_manuscripts](https://huggingface.co/magistermilitum/YOLO_manuscripts) | MIT | Segmentation layout |
| Kraken BLLA | [mittagessen/kraken](https://github.com/mittagessen/kraken) | Apache 2.0 | Segmentation lignes |

## Citation

```bibtex
@misc{cremma-mss-17,
  title     = {CREMMA HTR GT for 17th century MSS},
  author    = {HTR-United},
  year      = {2024},
  publisher = {HTR-United},
  url       = {https://github.com/HTR-United/CREMMA-MSS-17},
  license   = {CC-BY 4.0}
}
```

*Toutes les sources listées sont compatibles avec un usage de recherche non commerciale.*
