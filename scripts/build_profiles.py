import argparse
import json
from pathlib import Path

from hra.data import ASPECTS, read_jsonl
from hra.profiles import build_profiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, default=Path("data/processed/corpus_aspects.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/hotel_profiles.json"))
    ap.add_argument("--m", type=float, default=3.0)
    args = ap.parse_args()

    profiles = build_profiles(read_jsonl(args.inp), m=args.m)
    args.out.write_text(json.dumps(profiles, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    lines = [
        "# Hotel profiles",
        "",
        f"{len(profiles)} hotels, scores shrunk towards the city average with m = {args.m:g}.",
        "",
        "| aspect | hotels with >= 1 mention | hotels with >= 3 mentions | positive share |",
        "|---|---|---|---|",
    ]
    for a in ASPECTS:
        mentions = [p["aspects"][a]["mentions"] for p in profiles]
        pos = sum(p["aspects"][a]["positive"] for p in profiles)
        lines.append(
            f"| {a} | {sum(x >= 1 for x in mentions)} | {sum(x >= 3 for x in mentions)} | "
            f"{pos / max(sum(mentions), 1):.0%} |"
        )
    lines.append("")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/profiles.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
