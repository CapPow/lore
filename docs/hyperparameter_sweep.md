# LORE Hyperparameter Sweep -- Architecture Selection Notes

This document records the hyperparameter sweep conducted to select default
architecture settings for `lore/model.py`. It is intended as a reference for
methods writing and for users who wish to understand the rationale behind the
defaults or adapt them for new datasets.

---

## Background

LORE's disambiguation network (`LoreNet`) has several architectural knobs that
meaningfully affect performance: the base hidden dimension, the depth of the
stream encoders, and the dropout rate. Prior to this sweep, defaults were set
conservatively (`hidden_dim=256`, `encoder_depth=2`, `soil_encoder_depth=2`,
`dropout=0.1`) without systematic validation.

A key structural concern motivating the sweep was the soil stream encoder.
The soil feature block is a 123-dimensional probability vector (USDA soil great
group probabilities from Hengl & Nauman 2018). With `encoder_depth=2`, this
block was compressed from 123 dimensions to the stream output in a single
hidden layer -- a potentially lossy bottleneck given the feature count. A
dedicated `soil_encoder_depth` parameter was introduced to allow the soil
stream to be deeper than the numeric and date streams independently.

All sweep runs excluded `feat_lat` and `feat_lon` to assess discriminability
independent of geography. Latitude and longitude carry strong signal that
partially approximates the range shapefiles directly; including them would
likely dominate gradients and mask differences between architectural
configurations. The full-feature setting (including geography) is used in
production runs and is expected to perform at least as well.

---

## Sweep Design

Five configurations were tested, varying hidden dimension, encoder depth, soil
encoder depth, and dropout. All other hyperparameters were held at their
defaults (lr=5e-4, patience=50, max_epochs=500, batch_size=256). Stratified
splits used `random_state=42` and were recomputed identically for each run
from the same features parquet, so train/val/test membership was consistent
across configurations.

| Config | hidden_dim | encoder_depth | soil_encoder_depth | dropout | Notes |
|--------|-----------|--------------|-------------------|---------|-------|
| A      | 256        | 2             | 2                  | 0.1     | True baseline (old defaults) |
| B      | 256        | 2             | 3                  | 0.1     | Soil encoder fix only |
| C      | 512        | 2             | 3                  | 0.1     | Larger hidden dim |
| D      | 512        | 3             | 4                  | 0.1     | Deeper encoders |
| E      | 512        | 3             | 4                  | 0.25    | Higher dropout |

---

## Dataset 1 -- Peromyscus maniculatus complex (~86k records, 6 classes)

**GBIF DOI:** https://doi.org/10.15468/dl.3cv9hy
*(GBIF.org, 25 March 2026)*

Source taxon: *Peromyscus maniculatus*. Destination taxa: *P. maniculatus*,
*P. sonoriensis*, *P. gambelii*, *P. keeni*, *P. labecula*, *P. arcticus*.
This dataset has a substantial parapatric zone with 1,402 records flagged as
geospatially ambiguous, making it the primary dataset for evaluating
disambiguation quality.

| Config | hidden_dim | enc_depth | soil_depth | dropout | epochs | val_loss | test_acc | para_rate | n_params |
|--------|-----------|-----------|-----------|---------|--------|----------|----------|-----------|----------|
| A      | 256        | 2         | 2          | 0.10    | 165    | 0.013748 | 0.9941   | 1.0       | 388,422  |
| B      | 256        | 2         | 3          | 0.10    | 212    | 0.006713 | 0.9984   | 1.0       | 392,582  |
| C      | 512        | 2         | 3          | 0.10    | 158    | 0.007082 | 0.9961   | 1.0       | 1,497,862|
| D      | 512        | 3         | 4          | 0.10    | 294    | 0.003115 | 0.9989   | 1.0       | 1,547,398|
| E      | 512        | 3         | 4          | 0.25    | 241    | 0.006128 | 0.9958   | 1.0       | 1,547,398|

**Findings:** Config D produced the best validation loss (0.003115) and test
accuracy (99.9%) by a clear margin. The improvement from A to B confirms that
the deeper soil encoder alone provides a meaningful benefit. The further
improvement from B to D indicates that additional encoder depth and hidden
dimension contribute on top of the soil fix. Config E demonstrates that
`dropout=0.25` is harmful at this dataset size -- the model is not overfitting
badly enough to benefit from stronger regularization, and the higher dropout
rate degrades both val_loss and test accuracy relative to D.

The `para_rate` metric (fraction of parapatric records for which the top
prediction is one of the known candidate taxa) was 1.0 for all configurations.
This metric has a ceiling effect: without authoritative ground-truth labels for
parapatric records, any reasonable model will achieve full candidate-set
coverage. It is retained as a sanity check rather than a discriminating signal.
The distribution of predictions across parapatric records (i.e., which taxon
each ambiguous record is assigned to) is the more informative quantity for
downstream use, and is available in the full pipeline output.

---

## Dataset 2 -- Chaetodipus nelsoni complex (~5k records, 3 classes)

**GBIF DOI:** https://doi.org/10.15468/dl.cdmqxu
*(GBIF.org, 28 March 2026)*

Source taxon: *Chaetodipus nelsoni*. Destination taxa: *C. nelsoni*,
*C. durangae*, *C. collis*. This dataset has no parapatric records, so
`para_rate` is not applicable. It represents the small-dataset regime and
was used to test whether the architecture selected for Peromyscus generalizes
or shows signs of overparameterization at reduced sample size.

Approximate training set size after stratified split: ~3,500 records.

| Config | hidden_dim | enc_depth | soil_depth | dropout | epochs | val_loss | test_acc | para_rate | n_params |
|--------|-----------|-----------|-----------|---------|--------|----------|----------|-----------|----------|
| A      | 256        | 2         | 2          | 0.10    | 142    | 0.025132 | 0.9797   | N/A       | 388,035  |
| B      | 256        | 2         | 3          | 0.10    | 132    | 0.028215 | 0.9730   | N/A       | 392,195  |
| C      | 512        | 2         | 3          | 0.10    | 137    | 0.023040 | 0.9797   | N/A       | 1,497,091|
| D      | 512        | 3         | 4          | 0.10    | 108    | 0.041394 | 0.9764   | N/A       | 1,546,627|
| E      | 512        | 3         | 4          | 0.25    | 157    | 0.018679 | 0.9730   | N/A       | 1,546,627|

**Findings:** The ranking reverses substantially relative to the Peromyscus
results. Config D, the clear winner at 86k records, produces the worst
validation loss at 5k (0.041), and its early stopping at 108 epochs suggests
unstable training consistent with overparameterization. Config E recovers
substantially with `dropout=0.25` (val_loss 0.019), confirming that the larger
architecture remains viable at small dataset sizes when regularization is
increased. Configs A and C are competitive, with C achieving marginally better
val_loss despite having nearly 4x the parameters of A.

---

## Interpretation and Recommendations

The two sweeps together reveal a clear dataset-size dependency in the optimal
dropout rate, with the underlying architecture otherwise stable.

**Architecture defaults (all dataset sizes):**

```
hidden_dim         = 512
encoder_depth      = 3
soil_encoder_depth = 4
decoder_depth      = 3   (unchanged from original)
```

**Dropout guidance:**

The optimal dropout rate is inversely correlated with dataset size. Based on
the two datasets tested:

| Approximate training records | Recommended dropout |
|------------------------------|---------------------|
| ~3,500 (small)               | 0.25                |
| ~60,000+ (large)             | 0.10                |

Users working with intermediate dataset sizes should treat dropout as the
primary tuning knob. The default in `lore/model.py` is `dropout=0.1`, which
is appropriate for large datasets. For smaller datasets (roughly under ~10k
training records), increasing dropout to 0.2-0.3 is advised. The
`--dropout` flag in both `lore/model.py` and `run_pipeline.py` exposes this
parameter directly.

Note that only two data points are available for this relationship. A more
precise dropout schedule would require additional datasets of intermediate
size. Users with access to such datasets are encouraged to run the provided
`sweep.py` script (see below) and contribute findings.

---

## Caveats

- All runs excluded `feat_lat` and `feat_lon`. Performance in the full-feature
  setting (the production default) is expected to be at least as good, as
  geography provides strong additional signal.

- The `para_rate` metric does not constitute ground-truth validation of
  parapatric disambiguation. It measures only whether the model's top
  prediction falls within the set of geospatially plausible candidates, not
  whether the assignment is taxonomically correct. Authoritative validation
  would require vouchered specimens with confirmed identifications from the
  parapatric zone.

- The Chaetodipus dataset has no parapatric records, limiting its utility for
  evaluating disambiguation quality directly. It is used here solely to
  characterize architectural behavior at small dataset sizes.

- Parameter counts differ slightly between the two datasets (e.g., A: 388,422
  vs 388,035) due to differences in `n_vocab` (the verbatim name encoder
  vocabulary size), which scales with the number of distinct source taxon name
  strings in the dataset.

---

## Reproducing the Sweep

The sweep script used to generate these results is archived at
`docs/sweep.py`. It requires a completed feature extraction step (pipeline
steps 1-4) for the target dataset. To reproduce:

```bash
# Edit DEFAULT_FEATURES and BASE_RUN_TAG at the top of docs/sweep.py,
# then run from the project root:
python docs/sweep.py
```

Dependencies are identical to the main LORE requirements. Each configuration
writes a self-contained checkpoint to `runs/<BASE_RUN_TAG>_<label>/cache/model/`
for post-hoc inspection.

---

## References

Hengl T, Nauman T (2018). Predicted USDA soil great groups at 250 m
(probabilities). Zenodo. https://doi.org/10.5281/zenodo.3528062
