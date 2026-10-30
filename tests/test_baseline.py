import numpy as np

from hra.baseline import TfidfAspectModel


def test_learns_an_obvious_pattern():
    texts = ["kamar bersih sekali", "kamar kotor sekali", "staf ramah", "lokasi jauh"] * 5
    y = np.random.default_rng(0).integers(0, 4, size=(len(texts), 7))
    y[:, 1] = [1, 2, 0, 0] * 5  # Kebersihan: bersih -> positive, kotor -> negative

    model = TfidfAspectModel(C=10)
    model.words.min_df = model.chars.min_df = 1
    pred = model.fit(texts, y).predict(["kamar bersih sekali", "kamar kotor sekali"])

    assert pred.shape == (2, 7)
    assert pred[0, 1] == 1
    assert pred[1, 1] == 2
