import argparse
import json
from pathlib import Path

import joblib

from hra.data import load_split, read_jsonl, to_matrix
from hra.metrics import evaluate, majority_vote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--tfidf", type=Path, default=Path("models/tfidf_baseline.joblib"))
    ap.add_argument("--indobert", type=Path, default=Path("models/indobert"))
    args = ap.parse_args()

    test = load_split("test", args.data)
    ids = [r["review_id"] for r in test]
    texts = [r["text"] for r in test]
    y_gold = to_matrix(test)

    draft = {r["review_id"]: r for r in read_jsonl(args.data / "annotation_prefilled_gold500.jsonl")}
    blind = [{r["review_id"]: r for r in read_jsonl(args.data / f"annotator_{k}_blind.jsonl")} for k in (1, 2, 3)]

    preds = {"Gemini Flash (dataset draft)": to_matrix([draft[i] for i in ids])}
    if args.tfidf.exists():
        preds["TF-IDF + LR"] = joblib.load(args.tfidf).predict(texts)
    if args.indobert.exists():
        from hra.indobert import AspectModel, predict

        model, tokenizer, max_length = AspectModel.load(args.indobert)
        preds["IndoBERT"] = predict(model, tokenizer, texts, max_length)[0]

    rows_in_blind = [n for n, i in enumerate(ids) if i in blind[0]]
    blind_ids = [ids[n] for n in rows_in_blind]
    y_blind, agreed = majority_vote([to_matrix([b[i] for i in blind_ids]) for b in blind])
    keep = agreed.all(axis=1)
    sub = [n for n, k in zip(rows_in_blind, keep, strict=True) if k]
    y_blind, y_sub = y_blind[keep], y_gold[sub]

    results = {"Human, blind majority": {"blind_vs_gold": evaluate(y_sub, y_blind)["pair"]["f1"]}}
    for name, p in preds.items():
        full = evaluate(y_gold, p)
        results[name] = {
            "pair_f1": full["pair"]["f1"],
            "detection_f1": full["detection"]["f1"],
            "subset_vs_gold": evaluate(y_sub, p[sub])["pair"]["f1"],
            "subset_vs_blind": evaluate(y_blind, p[sub])["pair"]["f1"],
        }

    lines = [
        "# Model comparison",
        "",
        f"Gold test: {len(ids)} reviews. Blind subset: {len(sub)} of them re-labelled without the draft.",
        "",
        "| Model | Pair F1 (gold) | Detection F1 (gold) | Pair F1 subset vs gold | Pair F1 subset vs blind |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        if "pair_f1" in r:
            row = [r["pair_f1"], r["detection_f1"], r["subset_vs_gold"], r["subset_vs_blind"]]
        else:
            row = ["-", "-", r["blind_vs_gold"], 1.0]
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/comparison.md").write_text("\n".join(lines), encoding="utf-8")
    Path("reports/comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
