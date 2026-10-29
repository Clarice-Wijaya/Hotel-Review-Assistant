# Hotel Review Assistant

## Data

I use [IndoHotelABSA](https://data.mendeley.com/datasets/9vhzg5wkf9), a set of 14,988 Google Maps reviews of 3,075 hotels in 55 Indonesian cities. Each labelled review says which of 7 aspects it mentions (location, cleanliness, service, room, price, food, public facilities) and whether the mention is positive, negative or neutral.

- **train**: 2,480 reviews labelled by an LLM and not checked by people (silver)
- **val / test**: 100 / 400 reviews checked by three annotators (gold)

I train on silver and evaluate on gold only.

The review text belongs to Google Maps and may only be used for non-commercial research,
so the data is not in this repo. Download it into `data/raw/` to run the code.

A few things from the [EDA](reports/eda.md):

- Neutral is only 4.5% of test labels, so models will struggle with it.
- Location is almost always positive; cleanliness is never neutral.
- Every hotel has at most 5 reviews, so hotel-level scores need smoothing.
