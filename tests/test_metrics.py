import numpy as np

from hra.data import to_matrix
from hra.metrics import evaluate


def rec(*labels):
    return {"text": "x", "labels": [{"aspect": a, "polarity": p} for a, p in labels]}


def test_perfect_prediction():
    y = to_matrix([rec(("Lokasi", "POSITIF")), rec(("Harga", "NETRAL"), ("Kebersihan", "NEGATIF"))])
    m = evaluate(y, y)
    assert m["pair"]["f1"] == 1.0
    assert m["detection"]["f1"] == 1.0


def test_predicting_nothing_scores_zero():
    y = to_matrix([rec(("Lokasi", "POSITIF"))])
    assert evaluate(y, np.zeros_like(y))["pair"]["f1"] == 0.0


def test_right_aspect_wrong_polarity():
    y = to_matrix([rec(("Lokasi", "POSITIF"))])
    pred = to_matrix([rec(("Lokasi", "NEGATIF"))])
    m = evaluate(y, pred)
    assert m["detection"]["f1"] == 1.0
    assert m["pair"]["f1"] == 0.0


def test_half_right():
    y = to_matrix([rec(("Lokasi", "POSITIF"), ("Harga", "NEGATIF"))])
    pred = to_matrix([rec(("Lokasi", "POSITIF"), ("Pelayanan", "POSITIF"))])
    m = evaluate(y, pred)["pair"]
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
