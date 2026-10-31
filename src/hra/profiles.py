import hashlib
from collections import defaultdict

from hra.data import ASPECTS

SCORE = {"POSITIF": 1.0, "NEGATIF": 0.0, "NETRAL": 0.5}


def hotel_id(city, name):
    return "h_" + hashlib.sha1(f"{city}::{name}".encode()).hexdigest()[:10]


def city_priors(rows, min_mentions=20):
    """Average sentiment per (city, aspect). Cities with few mentions fall back to the national average."""
    total, n = defaultdict(float), defaultdict(int)
    for r in rows:
        for a, lab in r["aspects"].items():
            for key in [(r["city"], a), ("*", a)]:
                total[key] += SCORE[lab["polarity"]]
                n[key] += 1
    national = {a: total["*", a] / n["*", a] if n["*", a] else 0.5 for a in ASPECTS}
    priors = {}
    for (city, a), count in n.items():
        if city != "*":
            priors[city, a] = total[city, a] / count if count >= min_mentions else national[a]
    return priors, national


def build_profiles(rows, m=3.0):
    priors, national = city_priors(rows)
    by_hotel = defaultdict(list)
    for r in rows:
        by_hotel[r["city"], r["hotel"]].append(r)

    profiles = []
    for (city, name), reviews in by_hotel.items():
        aspects = {}
        for a in ASPECTS:
            hits = [(r["review_id"], r["aspects"][a]) for r in reviews if a in r["aspects"]]
            prior = priors.get((city, a), national[a])
            total = sum(SCORE[lab["polarity"]] for _, lab in hits)
            pos = sorted((lab["confidence"], rid) for rid, lab in hits if lab["polarity"] == "POSITIF")
            neg = sorted((lab["confidence"], rid) for rid, lab in hits if lab["polarity"] == "NEGATIF")
            aspects[a] = {
                "mentions": len(hits),
                "positive": len(pos),
                "negative": len(neg),
                "score": round((total + m * prior) / (len(hits) + m), 4),
                "city_prior": round(prior, 4),
                "best_positive": pos[-1][1] if pos else None,
                "best_negative": neg[-1][1] if neg else None,
            }
        ratings = [r["rating"] for r in reviews]
        profiles.append(
            {
                "hotel_id": hotel_id(city, name),
                "name": name,
                "city": city,
                "n_reviews": len(reviews),
                "mean_rating": round(sum(ratings) / len(ratings), 2),
                "aspects": aspects,
            }
        )
    return sorted(profiles, key=lambda p: (p["city"], p["name"]))
