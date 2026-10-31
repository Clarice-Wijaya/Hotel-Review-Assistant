# Hotel Review Assistant

Search hotels in Indonesia by what past guests actually said. Reviews are turned into
per-aspect sentiment (cleanliness, service, location, room, price, food, public facilities),
so a request like *"hotel bersih dekat pusat kota, pelayanannya ramah"* can be matched to
hotels and the reviews behind them.

Work in progress: aspect sentiment model done; hotel profiles, search and assistant next.


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


## Results

All scores are on the 400 gold test reviews. Pair F1 counts a prediction as right only
when both the aspect and its sentiment are right.

| Model | Pair F1 | Detection F1 | Speed |
|---|---|---|---|
| Predict "not mentioned" everywhere | 0.000 | 0.000 | - |
| TF-IDF + logistic regression | 0.731 | 0.843 | 0.5 ms/review (CPU) |
| IndoBERT fine-tuned (3 seeds) | **0.771 ± 0.007** | 0.883 | 16 ms/review (T4 GPU) |

IndoBERT helps most where the wording varies: public facilities (+0.09), price (+0.08)
and food (+0.07). For cleanliness and room it barely beats TF-IDF, since words like
*bersih* and *kotor* already carry most of the signal.

Validation F1 stops improving after epoch 4-5 while training loss keeps falling, so the
epoch is picked on validation, not taken from the last epoch.

Details: [reports/baseline_tfidf.json](reports/baseline_tfidf.json),
[reports/indobert.json](reports/indobert.json).


   ### How fair is the gold test?

   The training labels were drafted by Gemini Flash, and the gold labels were made by people
   correcting that draft. 100 test reviews were also re-labelled blind (no draft shown).
   On those reviews ([reports/comparison.md](reports/comparison.md)):

   | Model | Pair F1 vs gold | Pair F1 vs blind |
   |---|---|---|
   | Gemini Flash (the dataset's draft) | 0.884 | 0.755 |
   | IndoBERT (this repo) | 0.786 | 0.717 |
   | TF-IDF baseline | 0.746 | 0.688 |

   Gemini drops the most, because gold started from its own answers. Against blind labels the
   gap between Gemini and IndoBERT shrinks from 0.10 to 0.04, while IndoBERT runs locally
   with no API cost. With only 97 reviews, differences of a few points are within noise.


## Run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
# put the IndoHotelABSA .jsonl files in data/raw/
pytest -q
python scripts/eda.py
python scripts/train_baseline.py
```

IndoBERT needs a GPU: open `notebooks/train_indobert.ipynb` in Colab.
