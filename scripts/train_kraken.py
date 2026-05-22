"""
scripts/train_kraken.py
------------------------
Fine-tune un modèle Kraken sur le corpus CREMMA Medieval.

Usage:
    python scripts/train_kraken.py
    python scripts/train_kraken.py --base-model catmus-medieval --epochs 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.htr.kraken_htr import fine_tune


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning Kraken")
    p.add_argument("--train",      default="data/splits/train.json")
    p.add_argument("--val",        default="data/splits/val.json")
    p.add_argument("--base-model", default="catmus-medieval",
                   help="Identifiant HF ou chemin local (.mlmodel)")
    p.add_argument("--output-dir", default="models/kraken_cremma_v1")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"🏋️  Fine-tuning Kraken")
    print(f"   Base model   : {args.base_model}")
    print(f"   Epochs       : {args.epochs}")
    print(f"   Batch size   : {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Output dir   : {args.output_dir}\n")

    fine_tune(
        train_manifest=args.train,
        val_manifest=args.val,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
