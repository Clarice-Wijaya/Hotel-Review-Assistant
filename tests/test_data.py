from hra.data import ASPECTS, to_matrix


def rec(*labels):
    return {"text": "x", "labels": [{"aspect": a, "polarity": p} for a, p in labels]}


def test_to_matrix_puts_labels_in_the_right_cells():
    y = to_matrix([rec(("Lokasi", "POSITIF"), ("Harga", "NEGATIF"))])

    assert y.shape == (1, 7)
    assert y[0, ASPECTS.index("Lokasi")] == 1
    assert y[0, ASPECTS.index("Harga")] == 2
    assert (y == 0).sum() == 5


def test_unknown_aspect_or_polarity_is_ignored():
    y = to_matrix([rec(("Parkir", "POSITIF")), rec(("Lokasi", "MAYBE"))])

    assert y.shape == (2, 7)
    assert (y == 0).all()


def test_review_without_labels_is_all_zero():
    y = to_matrix([{"text": "review dari corpus, tanpa label"}])

    assert (y == 0).all()
