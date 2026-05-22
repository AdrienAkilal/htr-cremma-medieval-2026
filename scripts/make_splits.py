"""
scripts/make_splits.py
-----------------------
Constitue les splits train / val / test depuis le manifest CREMMA-MSS-17.

Stratégie : split par DOCUMENT (xml_path), pas par ligne.
Cela évite toute fuite de données entre les splits.

  train : 80 % des documents
  val   : 10 % des documents
  test  : 10 % des documents  ← scellé avec SHA-256

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --manifest data/raw/cremma_mss17_manifest.json
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED       = 42
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# TEST_FRAC = 0.10  (le reste)


def make_splits(manifest_path: str,
                output_dir: str = "data/splits",
                seed: int = SEED) -> tuple[list, list, list]:
    """Partitionne le corpus par document en train / val / test.

    Le split est fait au niveau du document (xml_path) pour éviter
    toute fuite entre les ensembles. Toutes les lignes d'un même
    document restent dans le même split.

    Args:
        manifest_path: Chemin vers le manifest JSON (sortie de download_corpus).
        output_dir: Dossier de sortie des splits.
        seed: Seed pour la reproductibilité (défaut 42).

    Returns:
        Tuple (train, val, test) — listes de dicts (niveau ligne).

    Raises:
        FileNotFoundError: Si manifest_path est introuvable.
    """
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Manifest introuvable : {manifest_path}")

    random.seed(seed)

    records = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    # Grouper par document
    docs = defaultdict(list)
    for r in records:
        docs[r["xml_path"]].append(r)

    doc_keys = sorted(docs.keys())
    random.shuffle(doc_keys)

    n       = len(doc_keys)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_docs = doc_keys[:n_train]
    val_docs   = doc_keys[n_train:n_train + n_val]
    test_docs  = doc_keys[n_train + n_val:]

    # Aplatir en lignes
    train = [r for k in train_docs for r in docs[k]]
    val   = [r for k in val_docs   for r in docs[k]]
    test  = [r for k in test_docs  for r in docs[k]]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n📊  Split (seed={seed}) — {n} documents, {len(records)} lignes")
    print(f"┌─────────┬───────────┬────────┐")
    print(f"│  Split  │  Docs     │ Lignes │")
    print(f"├─────────┼───────────┼────────┤")

    for name, split, n_docs in [
        ("train", train, len(train_docs)),
        ("val",   val,   len(val_docs)),
        ("test",  test,  len(test_docs)),
    ]:
        path = out / f"{name}.json"
        path.write_text(
            json.dumps(split, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"│  {name:<7}│  {n_docs:>7}  │ {len(split):>6} │")

    print(f"└─────────┴───────────┴────────┘")

    # Scellement du test set
    test_bytes  = (out / "test.json").read_bytes()
    sha256_test = hashlib.sha256(test_bytes).hexdigest()
    (out / "test_sha256.txt").write_text(sha256_test, encoding="utf-8")

    print(f"\n🔒  Test set scellé")
    print(f"   SHA-256 : {sha256_test}")
    print(f"   → Copiez ce hash dans votre README !\n")

    # Log sous-corpus dans chaque split
    _print_subcorpus_distribution(train, val, test)

    return train, val, test


def _print_subcorpus_distribution(train, val, test):
    """Affiche la répartition des sous-corpus dans chaque split."""
    from collections import Counter
    print("Répartition des sous-corpus :")
    all_subs = sorted({r["subcorpus"] for r in train + val + test})
    print(f"  {'Sous-corpus':<45}  {'train':>6}  {'val':>6}  {'test':>6}")
    print(f"  {'-'*45}  {'------':>6}  {'------':>6}  {'------':>6}")
    for sub in all_subs:
        n_tr = sum(1 for r in train if r["subcorpus"] == sub)
        n_va = sum(1 for r in val   if r["subcorpus"] == sub)
        n_te = sum(1 for r in test  if r["subcorpus"] == sub)
        print(f"  {sub:<45}  {n_tr:>6}  {n_va:>6}  {n_te:>6}")


def parse_args():
    p = argparse.ArgumentParser(description="Création des splits CREMMA-MSS-17")
    p.add_argument("--manifest", default="data/raw/cremma_mss17_manifest.json")
    p.add_argument("--output",   default="data/splits")
    p.add_argument("--seed",     type=int, default=SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_splits(args.manifest, args.output, args.seed)
