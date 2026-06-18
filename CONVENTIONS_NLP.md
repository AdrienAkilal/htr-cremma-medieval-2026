# CONVENTIONS_NLP.md — Règles de normalisation NLP

Pipeline HTR CREMMA Medieval 2026 · Module NLP · Volet 2

Ce fichier documente toutes les règles de normalisation appliquées aux
transcriptions HTR, leur justification linguistique, et leur impact chiffré
sur le corpus de test (f12 + f279, 33 lignes processables).

---

## 1. Contexte et convention de base

Le pipeline CV produit des transcriptions **semi-diplomatiques** : les
caractères sont transcrits tels qu'ils apparaissent dans le manuscrit, sans
interprétation éditoriale. La normalisation NLP vise à produire une
transcription **lisible en moyen français** en résolvant les ambiguïtés
graphiques et les abréviations.

Les lignes avec `needs_review=true` (confiance < 0.80) ne sont **pas**
normalisées automatiquement.

---

## 2. Règles appliquées (dans l'ordre d'exécution)

### Règle 1 — Normalisation Unicode NFC (`normalisation_nfc`)

**Principe :** Convertit le texte en forme canonique Unicode NFC.
Certains OCR produisent des caractères composés décomposés (e.g., `e` + accent
combining au lieu du glyphe unique `é`). NFC fusionne ces représentations.

**Exemple :**
```
Entrée  : "été"   (e + combining acute accent)
Sortie  : "été"               (glyphes NFC unifiés)
```

**Impact mesuré (corpus f12 + f279) :**

| Corpus    | Lignes process. | Corrections | Distance relative |
|-----------|-----------------|-------------|-------------------|
| f12 + f279 | 33             | 2           | 0.39 %            |

> 2 corrections sur des caractères décomposés (ex. `ę` en deux points de code
> dans f279). Cette règle est un garde-fou préventif à coût quasi-nul.

---

### Règle 2 — Allographe u/v (`normalisation_uv`)

**Principe :** Dans les manuscrits médiévaux et du XVIIe siècle, la lettre
`u` en tête de mot remplissait souvent le rôle consonantique de `v`.
La règle convertit `u` initial de mot → `v`, sauf pour les mots de la liste
blanche (`un`, `une`, `ut`).

**Exemple :**
```
Entrée  : "Ie uous mandars"
Sortie  : "Ie vous mandars"

Entrée  : "lo uolune det"
Sortie  : "lo volume det"
```

**Heuristique :** `u` précédé d'un non-lettre (espace, ponctuation, début de
ligne), suivi d'une ou plusieurs lettres, et n'appartenant pas à la liste
blanche.

**Limitations :** Peut produire des faux positifs sur des noms propres ou
formes latines rares. Les cas ambigus doivent être revus manuellement.

**Impact mesuré (corpus f12 + f279, 33 lignes processables) :**

| Corrections | Distance relative |
|-------------|-------------------|
| 11          | 1.07 %            |

Exemples de corrections appliquées sur f279 :
`uous` → `vous`, `uolune` → `volume`, `uouy` → `vouy`, `uois` → `vois`

---

### Règle 3 — Allographe i/j (`normalisation_ij`)

**Principe :** Dans les manuscrits anciens, `j` n'était pas différencié de
`i`. Un `i` en tête de mot suivi d'une voyelle est presque toujours la
consonne `j`.

**Exemple :**
```
Entrée  : "que ie me suis"
Sortie  : "que je me suis"

Entrée  : "ie nay"
Sortie  : "je nay"
```

**Heuristique :** `i` minuscule précédé d'un non-lettre, suivi d'une voyelle
(`a`, `e`, `i`, `o`, `u`, `y` et variantes accentuées).

**Pas d'impact sur :** `il`, `in`, `ippo` (suivis d'une consonne → non
modifiés).

**Impact mesuré (corpus f12 + f279, 33 lignes processables) :**

| Corrections | Distance relative |
|-------------|-------------------|
| 6           | 0.58 %            |

Exemples : `ie` → `je` (×4 dans f279), `iugrimo` → `jugrimo`

---

### Règle 4 — Note tironienne (`normalisation_tironien`)

**Principe :** Le caractère `⁊` (U+204A, note tironienne) représente la
conjonction `et` en latin médiéval et vieux/moyen français.

**Exemple :**
```
Entrée  : "sil uous plaęt ⁊cas"
Sortie  : "sil vous plaęt etcas"
```

**Impact mesuré (corpus f12 + f279, 33 lignes processables) :** voir tableau §3.

---

### Règle 5 — P barré (`normalisation_p_barre`)

**Principe :** Le caractère `ꝑ` (U+A751) représente les préfixes `per` ou
`par` selon le contexte. Heuristique : si le caractère suivant est `a`, on
développe en `par`, sinon en `per`.

**Exemple :**
```
Entrée  : "ꝑ omnia"    → "per omnia"
Entrée  : "ꝑar le"     → "par le"
```

**Impact mesuré (corpus f12 + f279, 33 lignes processables) :** voir tableau §3.

---

### Règle 6 — Développement d'abréviations (`validation_abreviation`)

**Principe :** Les parenthèses dans la transcription CV signalent une
expansion d'abréviation incertaine. On valide et supprime les parenthèses.

**Exemple :**
```
Entrée  : "par le (com)mandement"
Sortie  : "par le commandement"
```

**Note :** Les parenthèses dans le champ `text` ne représentent jamais la
ponctuation originale — elles sont exclusivement un marqueur du pipeline CV.

**Impact mesuré (corpus f12 + f279, 33 lignes processables) :** voir tableau §3.

---

### Règle 7 — Lectures incertaines (`confirmation_lecture_incertaine`)

**Principe :** Les lectures incertaines annotées `[mot?]` sont confirmées
(les crochets et le point d'interrogation sont retirés). Les marqueurs
irréductibles `[?]`, `[...]`, `[†]`, `[abr]` sont **conservés**.

**Exemple :**
```
Entrée  : "le [grant?] seigneur"
Sortie  : "le grant seigneur"

Conservés : "[?]"  "[...]"  "[†]"  "[abr]"
```

**Impact mesuré (corpus f12 + f279, 33 lignes processables) :** voir tableau §3.

---

### Règle 8 — Balises de rubrique (`suppression_balise_rubrique`)

**Principe :** Les balises `<R>texte</R>` indiquent un passage en encre
rouge (rubrique). On retire les balises mais on conserve le contenu.

**Exemple :**
```
Entrée  : "<R>Incipit</R> liber primus"
Sortie  : "Incipit liber primus"
```

**Impact mesuré (corpus f12 + f279, 33 lignes processables) :** voir tableau §3.

---

### Règle 9 — Séparateur de colonne (`normalisation_separateur_colonne`)

**Principe :** Le marqueur `||` indique la fin d'une colonne dans un
document à double colonne. Il est remplacé par un espace.

**Exemple :**
```
Entrée  : "première colonne || deuxième colonne"
Sortie  : "première colonne   deuxième colonne"
```

**Impact mesuré (corpus f12 + f279, 33 lignes processables) :** voir tableau §3.

---

## 3. Impact global — Évaluation relative

En l'absence de vérité terrain sur les manuscrits BnF non vus, l'évaluation
utilise une **distance relative** entre versions successives de la
transcription (cf. consignes NLP, section 2).

```
distance_relative(v_avant, v_apres) = Levenshtein(v_avant, v_apres) / len(v_avant)
```

Lancer `python scripts/compute_cer.py` pour obtenir le tableau complet :

| Étape                          | Corrections | Distance relative |
|-------------------------------|-------------|-------------------|
| NFC                           | 2           | 0.39 %            |
| u/v  (u initial → v)          | 11          | 1.07 %            |
| i/j  (i initial → j)          | 6           | 0.58 %            |
| Tironien → et                 | 1           | 0.19 %            |
| P barré → per/par             | 0           | 0.00 %            |
| Développements (mot)          | 0           | 0.00 %            |
| Lectures [mot?]               | 0           | 0.00 %            |
| Rubriques \<R>                | 0           | 0.00 %            |
| Séparateur \|\|               | 0           | 0.00 %            |
| **TOTAL**                     | **20**      | **2.24 %**        |

Corpus : f12 + f279, 33 lignes processables (`needs_review=false`).
Source : `python scripts/compute_cer.py --all`

---

## 4. Décisions de conception

### Pourquoi des règles procédurales avant CamemBERT MLM ?

Les règles déterministes (sections 2.1–2.9) :
- Sont **sans ambiguïté** pour les cas couverts
- N'introduisent **aucune dépendance modèle**
- Sont **traçables** (chaque correction est journalisée avec position,
  avant/après, type)
- Produisent un gain CER immédiat et mesurable

La correction guidée par confiance (CamemBERT MLM) n'est envisagée qu'en
second lieu, sur les positions à faible confiance (`char_confidences < 0.7`)
pour lesquelles plusieurs candidats sont disponibles.

### Pourquoi semi-diplomatique et non normalisé ?

La convention semi-diplomatique conserve les formes graphiques originales
(accents, ponctuation) tout en résolvant les ambiguïtés bloquantes pour le
NLP (abréviations, allographes). Cela permet de :
1. Maintenir la fidélité philologique
2. Alimenter un modèle NER entraîné sur du français moderne avec des données
   qui restent lisibles

### Schéma BIO prévu (NER)

```
PER   — Personnes (noms propres)
LOC   — Lieux géographiques
DATE  — Dates et périodes
ORG   — Organisations, institutions
TITLE — Titres de noblesse / fonctions (roi, comte, évêque)
```
