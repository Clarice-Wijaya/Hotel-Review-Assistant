import argparse
import json
from pathlib import Path

import torch

from hra.data import ASPECTS, CLASSES, load_split
from hra.indobert import AspectModel, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--model", type=Path, default=Path("models/indobert"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/corpus_aspects.jsonl"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, max_length = AspectModel.load(args.model, device)
    corpus = load_split("corpus", args.data)
    ids, probs = predict(model, tokenizer, [r["text"] for r in corpus], max_length, batch_size=64, device=device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r, row, p in zip(corpus, ids, probs, strict=True):
            aspects = {
                a: {"polarity": CLASSES[c], "confidence": round(float(p[j, c]), 3)}
                for j, (a, c) in enumerate(zip(ASPECTS, row, strict=True))
                if c != 0
            }
            out = {k: r[k] for k in ("review_id", "hotel", "city", "rating")}
            f.write(json.dumps({**out, "aspects": aspects}, ensure_ascii=False) + "\n")
    print(f"labelled {len(corpus)} reviews -> {args.out}")


if __name__ == "__main__":
    main()
