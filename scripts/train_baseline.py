import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from hra.baseline import TfidfAspectModel
from hra.data import ASPECTS, load_split, to_matrix
from hra.metrics import evaluate


def texts(records):
    return [r["text"] for r in records]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    train, val, test = (load_split(s, args.data) for s in ["train", "val", "test"])
    y_train, y_val, y_test = to_matrix(train), to_matrix(val), to_matrix(test)

    val_f1 = {}
    for C in [0.5, 1, 2, 4, 8, 16]:
        model = TfidfAspectModel(C).fit(texts(train), y_train)
        val_f1[C] = evaluate(y_val, model.predict(texts(val)))["pair"]["f1"]
        print(f"C={C}: val pair F1 {val_f1[C]}")
    best_C = max(val_f1, key=val_f1.get)

    model = TfidfAspectModel(best_C).fit(texts(train), y_train)
    start = time.perf_counter()
    pred = model.predict(texts(test))
    ms = (time.perf_counter() - start) * 1000 / len(test)

    result = evaluate(y_test, pred)
    result["per_aspect_pair_f1"] = {
        a: evaluate(y_test[:, [j]], pred[:, [j]])["pair"]["f1"] for j, a in enumerate(ASPECTS)
    }
    result["baseline_all_none_pair_f1"] = evaluate(y_test, np.zeros_like(y_test))["pair"]["f1"]
    result["best_C"] = best_C
    result["val_pair_f1_by_C"] = val_f1
    result["ms_per_review"] = round(ms, 3)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline_tfidf.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/tfidf_baseline.joblib")
    print(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
