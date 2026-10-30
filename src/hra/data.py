import json
from pathlib import Path

import numpy as np

ASPECTS = [
    "Lokasi",
    "Kebersihan",
    "Pelayanan",
    "Kamar & Fasilitas",
    "Harga",
    "Makanan & Minuman",
    "Fasilitas Pendukung",
]
CLASSES = ["N/A", "POSITIF", "NEGATIF", "NETRAL"]


def read_jsonl(file_path):
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_split(name, data_dir=Path("data/raw")):
    file_mapping = {
        "train": "train_silver.jsonl",
        "val": "val_gold.jsonl",
        "test": "test_gold.jsonl",
        "corpus": "dataset_merged.jsonl",
    }
    if name not in file_mapping:
        raise ValueError(f"name harus salah satu dari {list(file_mapping)}, bukan {name!r}")
    file_path = Path(data_dir) / file_mapping[name]
    return read_jsonl(file_path)


def to_matrix(records):
    y = np.zeros((len(records), len(ASPECTS)), dtype=int)
    for i, record in enumerate(records):
        for lab in record.get("labels", []):
            if lab["aspect"] in ASPECTS and lab["polarity"] in CLASSES:
                aspect_idx = ASPECTS.index(lab["aspect"])
                class_idx = CLASSES.index(lab["polarity"])
                y[i, aspect_idx] = class_idx
    return y
