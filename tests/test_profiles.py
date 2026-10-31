from hra.profiles import build_profiles, hotel_id


def review(rid, hotel, city="x", **aspects):
    labels = {a: {"polarity": p, "confidence": 0.9} for a, p in aspects.items()}
    return {"review_id": rid, "hotel": hotel, "city": city, "rating": 5, "aspects": labels}


def test_more_evidence_moves_further_from_the_city_average():
    rows = [review(i, f"other{i}", Kebersihan="NEGATIF") for i in range(30)]
    rows += [review(100, "one", Kebersihan="POSITIF")]
    rows += [review(200 + i, "five", Kebersihan="POSITIF") for i in range(5)]
    scores = {p["name"]: p["aspects"]["Kebersihan"] for p in build_profiles(rows)}

    prior = scores["one"]["city_prior"]
    assert prior < scores["one"]["score"] < scores["five"]["score"] < 1.0
    assert scores["five"]["mentions"] == 5


def test_no_mentions_means_city_average():
    rows = [review(i, f"h{i}", Lokasi="POSITIF") for i in range(25)] + [review(99, "quiet")]
    quiet = next(p for p in build_profiles(rows) if p["name"] == "quiet")
    lokasi = quiet["aspects"]["Lokasi"]
    assert lokasi["mentions"] == 0
    assert lokasi["score"] == lokasi["city_prior"]
    assert lokasi["best_positive"] is None


def test_same_name_in_two_cities_is_two_hotels():
    rows = [review(1, "Hotel Santika", city="batam"), review(2, "Hotel Santika", city="bekasi")]
    profiles = build_profiles(rows)
    assert len(profiles) == 2
    assert hotel_id("batam", "Hotel Santika") != hotel_id("bekasi", "Hotel Santika")
