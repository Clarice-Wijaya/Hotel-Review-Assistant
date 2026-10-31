# Model comparison

Gold test: 400 reviews. Blind subset: 97 of them re-labelled without the draft.

| Model | Pair F1 (gold) | Detection F1 (gold) | Pair F1 subset vs gold | Pair F1 subset vs blind |
|---|---|---|---|---|
| Human, blind majority | - | - | 0.7534 | 1.0 |
| Gemini Flash (dataset draft) | 0.8718 | 0.9418 | 0.8843 | 0.7554 |
| TF-IDF + LR | 0.7306 | 0.8429 | 0.7461 | 0.6881 |
| IndoBERT | 0.7768 | 0.8842 | 0.7861 | 0.7169 |
