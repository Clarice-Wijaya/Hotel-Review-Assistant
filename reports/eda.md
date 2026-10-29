# EDA

## Splits

| split | reviews | aspect labels | labels/review |
|---|---|---|---|
| train | 2480 | 7338 | 2.96 |
| val | 100 | 296 | 2.96 |
| test | 400 | 1210 | 3.02 |
| corpus | 14988 | - | - |

train = silver (LLM labels, not checked), val/test = gold (checked by 3 annotators).

## Gold test: polarity per aspect

| aspect | pos | neg | neu | total |
|---|---|---|---|---|
| Lokasi | 100 | 13 | 9 | 122 |
| Kebersihan | 63 | 115 | 0 | 178 |
| Pelayanan | 119 | 154 | 4 | 277 |
| Kamar & Fasilitas | 112 | 176 | 15 | 303 |
| Harga | 53 | 52 | 9 | 114 |
| Makanan & Minuman | 52 | 39 | 9 | 100 |
| Fasilitas Pendukung | 48 | 59 | 9 | 116 |

## Notes

- overlap train vs val/test: 0 reviews
- neutral is only 4.5% of test labels
- 3075 hotels, median 5 and max 5 reviews each
- 7 hotel names appear in more than one city
