import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from hra.data import ASPECTS, load_split, to_matrix


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def split_sizes(splits, corpus):
    rows = []
    for name, recs in splits.items():
        n = int((to_matrix(recs) > 0).sum())
        rows.append([name, len(recs), n, round(n / len(recs), 2)])
    rows.append(["corpus", len(corpus), "-", "-"])
    return md_table(["split", "reviews", "aspect labels", "labels/review"], rows)


def polarity_by_aspect(y):
    rows = []
    for j, aspect in enumerate(ASPECTS):
        c = Counter(y[:, j].tolist())
        rows.append([aspect, c[1], c[2], c[3], c[1] + c[2] + c[3]])
    return md_table(["aspect", "pos", "neg", "neu", "total"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("reports/eda.md"))
    args = ap.parse_args()

    splits = {s: load_split(s, args.data) for s in ["train", "val", "test"]}
    corpus = load_split("corpus", args.data)
    y_test = to_matrix(splits["test"])

    labels = y_test[y_test > 0]
    neutral = (labels == 3).mean()

    train_ids = {r["corpus_review_id"] for r in splits["train"]}
    eval_ids = {r["corpus_review_id"] for s in ["val", "test"] for r in splits[s]}

    # hotel names repeat across cities, so (city, name) is the key
    per_hotel = Counter((r["city"], r["hotel"]) for r in corpus)
    cities = defaultdict(set)
    for r in corpus:
        cities[r["hotel"]].add(r["city"])
    shared_names = sum(len(c) > 1 for c in cities.values())

    lines = [
        "# EDA",
        "",
        "## Splits",
        "",
        split_sizes(splits, corpus),
        "",
        "train = silver (LLM labels, not checked), val/test = gold (checked by 3 annotators).",
        "",
        "## Gold test: polarity per aspect",
        "",
        polarity_by_aspect(y_test),
        "",
        "## Notes",
        "",
        f"- overlap train vs val/test: {len(train_ids & eval_ids)} reviews",
        f"- neutral is only {neutral:.1%} of test labels",
        f"- {len(per_hotel)} hotels, median {statistics.median(per_hotel.values()):.0f} "
        f"and max {max(per_hotel.values())} reviews each",
        f"- {shared_names} hotel names appear in more than one city",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
