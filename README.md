# LORE: Latent Occurrence Resolution Engine

LORE (Latent Occurrence Resolution Engine) is a Python tool for taxonomic
disambiguation of occurrence records following taxonomic splits. Given a
set of destination taxon range maps and a corpus of source occurrence records
collected under a pre-split name, LORE assigns each record a suggested taxon
label via geospatial intersection, then trains a multi-input neural network
to resolve records that fall in regions of parapatry or outside all known
ranges.

The name is a reference to the Star Trek: The Next Generation android Data,
whose twin Lore serves as a conceptual parallel. Records that appear
identical on the surface resolve to different identities under scrutiny.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/CapPow/lore.git
cd lore
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare an occurrence dataset

LORE accepts either a GBIF occurrence download or a custom Darwin Core CSV
(see Custom occurrence files under Module Reference for format requirements).
For most use cases, a GBIF Simple download is recommended.

If you are following the examples below, the GBIF datasets have already been
prepared and their DOIs are ready to use directly. Skip to step 4.

To prepare a dataset for your own taxon:

1. Go to [gbif.org/occurrence/download](https://www.gbif.org/occurrence/download)
2. Add filters:
   - **Has coordinate:** true
   - **Scientific name:** your pre-split source taxon (e.g. *Peromyscus maniculatus*)
3. Select **Format: Simple** and request the download
4. Once the download is ready, GBIF assigns it a DOI (e.g. `10.15468/dl.3cv9hy`).
   Pass this DOI to `--gbif-doi` in the pipeline command below.

To see which MDD taxonomic groups are available for range map download before
running the pipeline:

```bash
python scripts/download_data.py --list-mdd-groups
```

### 4. Run the full pipeline

The pipeline handles data acquisition, geospatial disambiguation, feature
extraction, model training, inference, and figure generation in a single
command. Steps are skipped automatically if their outputs already exist.

```bash
# Peromyscus example (~86k records, 6 classes)
python run_pipeline.py \
    --run-tag peromyscus_split_2026 \
    --gbif-doi 10.15468/dl.3cv9hy \
    --source-taxa "Peromyscus maniculatus" \
    --dest-taxa "Peromyscus maniculatus" "Peromyscus sonoriensis" \
                "Peromyscus gambelii" "Peromyscus keeni" \
                "Peromyscus labecula" "Peromyscus arcticus" \
    --mdd-group Rodentia \
    --workers 8

# Chaetodipus example (~5k records, 3 classes; increased dropout for small datasets)
python run_pipeline.py \
    --run-tag chaetodipus_nelsoni_split_2026 \
    --gbif-doi 10.15468/dl.cdmqxu \
    --source-taxa "Chaetodipus nelsoni" \
    --dest-taxa "Chaetodipus nelsoni" "Chaetodipus durangae" "Chaetodipus collis" \
    --mdd-group Rodentia \
    --dropout 0.25 \
    --workers 8
```

On re-run after reviewing the analysis report (steps 1-5 skipped automatically).
Review `runs/<run-tag>/analysis/analysis_report.txt` and pass any FLAT-rated
features to `--exclude-features`. Lat/lon are shown here as an example; your
report may flag different features or none at all:

```bash
python run_pipeline.py \
    --run-tag peromyscus_split_2026 \
    --gbif-doi 10.15468/dl.3cv9hy \
    --source-taxa "Peromyscus maniculatus" \
    --dest-taxa "Peromyscus maniculatus" "Peromyscus sonoriensis" \
                "Peromyscus gambelii" "Peromyscus keeni" \
                "Peromyscus labecula" "Peromyscus arcticus" \
    --mdd-group Rodentia \
    --exclude-features feat_lat feat_lon \
    --force
```

### 5. Generate a figure

Figure generation runs automatically as step 8 of `run_pipeline.py`. To
generate a figure independently:

```bash
python -m lore.visualize \
    --disambiguated runs/peromyscus_split_2026/disambiguated.csv \
    --ranges-file lore/data/ranges/MDD_Rodentia.gpkg \
    --dest-taxa "Peromyscus maniculatus" "Peromyscus sonoriensis" \
                "Peromyscus gambelii" "Peromyscus keeni" \
                "Peromyscus labecula" "Peromyscus arcticus" \
    --output runs/peromyscus_split_2026/figures/map.png \
    --basemap-dir lore/data/basemap
```

---

## Pipeline Overview

```
occurrences.csv  +  lore/data/ranges/MDD_<Group>.gpkg
        |
        v
   lore/geo.py                  Geospatial disambiguation
        |                       (spatial intersection + uncertainty buffering)
        v
geo_disambiguated.parquet       suggested_names column assigned
        |
        +-- single-label ----------------------------> training data
        +-- ambiguous (parapatric + out_of_range) --> ML targets
        |
        v
scripts/preprocess_rasters.py   Clip global rasters to bbox, derive slope
        |
        v
runs/<run_tag>/cache/           Regional rasters + soil stats + land cover + bbox record
        |
        v
   lore/features.py             Raster sampling + feature assembly
        |
        v
runs/<run_tag>/features.parquet         149 feature columns per record
        |
        v
   lore/analysis.py             Feature discriminability analysis
        |                       (KW-H, MI, ANOVA-F, soil MANOVA)
        v
runs/<run_tag>/analysis/analysis_report.txt   Human-reviewed -> optional --exclude-features
        |
        v
   lore/model.py                Multi-input PyTorch network
        |                       (stratified split, inverse-freq weighting,
        |                        class-conditional imputation)
        v
runs/<run_tag>/cache/model/checkpoint.pt      Self-contained model checkpoint
        |
        v
   lore/predict.py              Inference on ambiguous records
        |
        v
runs/<run_tag>/disambiguated.csv              Final taxon assignments
        |
        v
   lore/visualize.py            Spatial figure generation (PNG, 300 DPI)
```

---

## Repository Structure

```
LORE/
├── lore/                           Core importable library
│   ├── __init__.py
│   ├── geo.py                      Geospatial disambiguation
│   ├── features.py                 Raster sampling and feature assembly
│   ├── analysis.py                 Feature discriminability analysis
│   ├── model.py                    Neural network definition and training
│   ├── predict.py                  Inference and final output production
│   ├── visualize.py                Spatial figure generation
│   └── data/
│       ├── rasters/                Global source rasters (download once)
│       │   ├── worldclim/          WorldClim 2.1 30s elevation + bioclim bands
│       │   └── soil/               USDA soil great group probability rasters
│       ├── ranges/                 MDD range map geopackages (download once per group)
│       │   └── MDD_<Group>.gpkg
│       └── basemap/                Natural Earth 1:10m vectors (download once)
│           ├── ne_10m_land/
│           ├── ne_10m_admin_0_countries/
│           └── ne_10m_admin_1_states_provinces/
├── scripts/
│   ├── download_data.py            One-time acquisition of all data
│   └── preprocess_rasters.py       Clip global rasters to bbox, derive slope
├── runs/                           Per-run outputs (one subdirectory per run-tag)
│   └── <run_tag>/
│       ├── occurrences.csv
│       ├── geo_disambiguated.parquet
│       ├── features.parquet
│       ├── disambiguated.csv
│       ├── analysis/
│       │   ├── analysis_report.txt
│       │   └── feature_stats.csv
│       ├── figures/
│       │   └── map.png
│       └── cache/                  Preprocessed raster cache for this run
│           ├── bbox.json
│           ├── elevation.tif
│           ├── slope.tif
│           ├── wc2.1_30s_bio_*.tif
│           ├── soil/               123 clipped soil tifs
│           ├── soil_class_names.json
│           ├── soil_stats.json
│           ├── taxon_name_encoder.json
│           └── model/
│               ├── checkpoint.pt
│               ├── training_log.csv
│               └── training_summary.txt
├── docs/
│   ├── gbif_datasets.txt           GBIF download DOIs for example datasets
│   ├── hpc_sol.md                  Guide for running LORE on HPC
│   ├── hyperparameter_sweep.md     Architecture selection notes and sweep results
│   └── sweep.py                    Hyperparameter sweep script (archived)
├── tests/
│   ├── test_geo.py
│   ├── test_features.py
│   └── test_preprocess_rasters.py
├── run_pipeline.py                 End-to-end pipeline orchestration
├── requirements.txt
└── README.md
```

---

## Module Reference

### `scripts/download_data.py`

One-time acquisition of all data required for a LORE run. Safe to re-run;
all downloads are skipped if output files already exist. Called automatically
by `run_pipeline.py`; invoke directly only when acquiring data independently
of the pipeline.

```
--list-mdd-groups               Print available MDD taxonomic groups and exit
--gbif-doi DOI                  GBIF occurrence download DOI or key
--occurrences-file PATH         Custom Darwin Core CSV (mutually exclusive with --gbif-doi)
--mdd-group GROUP               MDD taxonomic group (default: Rodentia)
--mdd-zenodo-record ID          Zenodo record ID for MDD range maps.
                                Default: 6644197 (concept DOI, resolves to latest
                                MDD version). Pass a versioned record ID to pin
                                to a specific release.
--output-dir PATH               Destination for run-specific files (occurrences).
                                Required when downloading occurrences. Not needed
                                if only downloading global data (--skip-occurrences).
--ranges-dir PATH               Destination for MDD range geopackages (shared across runs)
--raster-dir PATH               Destination for global rasters (shared across runs)
--basemap-dir PATH              Destination for Natural Earth vectors (shared across runs)
--skip-occurrences              Skip occurrence acquisition
--skip-ranges                   Skip MDD range map download
--skip-worldclim                Skip WorldClim download
--skip-soil                     Skip soil raster download
--skip-basemap                  Skip Natural Earth basemap download
```

**Custom occurrence files** must be Darwin Core compatible CSV or TSV with
these exact column names:

| Column | Requirement |
|---|---|
| `gbifID` | Unique record identifier (any string or integer) |
| `decimalLatitude` | Decimal degrees, EPSG:4326, range [-90, 90] |
| `decimalLongitude` | Decimal degrees, EPSG:4326, range [-180, 180] |
| `coordinateUncertaintyInMeters` | Metres; null values are imputed by median |
| `verbatimScientificName` | Must contain the pre-split source taxon name(s) |
| `eventDate` | ISO 8601 preferred (YYYY-MM-DD or YYYY); null accepted |

A lightweight preflight check (column presence, file integrity) runs at copy
time. Full validation (coordinate ranges, type coercion, taxon matching) runs
at pipeline runtime in `load_occurrences()`.

---

### `scripts/preprocess_rasters.py`

Clips global rasters to the bounding box of a set of occurrence records and
writes a run cache directory. Derives slope from elevation via numpy gradient.
Processes soil probability rasters individually to avoid loading the full
stack into RAM.

If a stored `bbox.json` differs from the current occurrence bbox, a warning
is raised naming the number of potentially stale tifs. The cache is not
automatically invalidated; pass `--force` to regenerate. `bbox.json` is
only written on a fresh run or when `--force` is passed. No-op runs (all
files already exist) do not overwrite the stored bbox.

```
--run-tag TAG           Identifier for this run
--occurrences PATH      Path to geo_disambiguated.parquet
--raster-dir PATH       Directory containing downloaded global rasters
--output-dir PATH       Base output directory. Cache written to
                        <output-dir>/<run-tag>/cache/ (default: runs/)
--buffer-deg FLOAT      Degrees of bbox padding (default: 1.0)
--workers INT           Concurrent I/O workers (default: 4)
--force                 Overwrite existing cached rasters
```

---

### `lore/geo.py`

Geospatial disambiguation. Assigns each occurrence a `suggested_names` label
via spatial intersection with destination range polygons. Each record's
coordinate uncertainty is used to construct a circular buffer representing
the collection region, sampled as a MultiPoint for spatial join efficiency.

An allopatry check runs automatically before disambiguation, printing a
pairwise range overlap table. A warning is raised if any pair exceeds
`--allopatry-threshold` (default 0.1). Pass `--strict-allopatry` to convert
the warning to an exception.

**Output labels:**
- Binomial name: record falls within exactly one range
- `name_a | name_b`: record overlaps multiple ranges (parapatric)
- `out_of_range`: record falls outside all ranges
- `excessive_uncertainty`: coordinate uncertainty exceeds ceiling (default 128,576 m)

**Public API:**
```python
from lore.geo import load_ranges, load_occurrences, disambiguate, describe_results, allopatry_report

ranges = load_ranges("lore/data/ranges/MDD_Rodentia.gpkg",
                     taxa=["Peromyscus maniculatus", ...])
occ    = load_occurrences("runs/peromyscus_split_2026/occurrences.csv",
                          source_taxa=["Peromyscus maniculatus"])
occ    = disambiguate(occ, ranges, processes=8)
describe_results(occ, ranges)
occ.to_parquet("runs/peromyscus_split_2026/geo_disambiguated.parquet", index=False)
```

```
--occurrences PATH              Path to GBIF occurrence CSV (Darwin Core format)
--ranges-file PATH              Path to MDD range map geopackage
--source-taxa TAXON [...]       Pre-split taxon name(s) to filter occurrences
--dest-taxa TAXON [...]         Post-split destination taxon names
--run-tag TAG                   Run identifier (used to derive default output path)
--output-dir PATH               Base output directory (default: runs/)
--output PATH                   Explicit output path (overrides --run-tag default)
--name-col STR                  Column in range file containing taxon names
                                (default: sciname)
--workers INT                   Parallel processes for disambiguation
--allopatry-threshold FLOAT     Max acceptable pairwise overlap ratio (default: 0.1)
--strict-allopatry              Raise exception if overlap exceeds threshold
```

---

### `lore/features.py`

Samples environmental rasters at occurrence coordinates and assembles a
feature parquet. Uses `rasterio.DatasetReader.sample()` for soil rasters
to avoid full-band reads.

**149 feature columns per record:**

*Numeric stream (9 features):*

| Feature | Source |
|---|---|
| `feat_lat` | Decimal latitude |
| `feat_lon` | Decimal longitude |
| `feat_elevation` | WorldClim 30s elevation (m) |
| `feat_slope` | Slope (degrees), derived from elevation |
| `feat_bio1` | Mean annual temperature (x10 degrees C) |
| `feat_bio4` | Temperature seasonality (SD x 100) |
| `feat_bio7` | Temperature annual range, BIO5-BIO6 (x10 degrees C) |
| `feat_bio12` | Annual precipitation (mm) |
| `feat_bio15` | Precipitation seasonality (CV) |

All numeric features are min-max normalised over the specific occurrence
records in each run. Values are not directly comparable across runs.

*Date stream (2 features):* `feat_sin_doy`, `feat_cos_doy` - cyclic encoding
of day-of-year. Missing `eventDate` sets both to 0.0. Date ranges spanning
more than 60 days are treated as missing; year-month partial dates impute
the day to the 15th.

*Soil stream (123 features):* `feat_soil_<classname>` - USDA soil great group
probability at occurrence coordinates (Hengl & Nauman 2018, 250 m resolution).
One feature per class, values in [0, 1].

*Land cover stream (12 features):* `feat_lc_<classname>` - EarthEnv consensus
land cover percentage cover at occurrence coordinates (Tuanmu & Jetz 2014, 1 km
resolution). One feature per class, values normalized to [0, 1]. Classes:
needleleaf trees, evergreen broadleaf trees, deciduous broadleaf trees,
mixed/other trees, shrubs, herbaceous, cultivated, flooded vegetation, urban,
snow/ice, barren, open water.

*Name stream (1 feature):* `feat_taxon_name_encoded` - integer-encoded taxon
name built from the GBIF `species` field joined with `infraspecificEpithet`
when present (e.g. `"Microtus arvalis orcadensis"`). Encoder fit on full dataset
and serialized to `taxon_name_encoder.json` in the run cache. Subspecific
epithets are retained as they frequently correspond to post-split destination
taxa and carry geographic signal. When only one unique name is present, the name
stream is automatically disabled during training.

**Note on soil sampling speed:** Sampling rasters is I/O bound, and can take 
up to 2 hours. One-time cost per run; output parquet is cached.

```
--occurrences PATH      Path to geo_disambiguated.parquet
--run-tag TAG           Run tag matching a preprocess_rasters.py cache
--output-dir PATH       Base output directory. Cache read from
                        <output-dir>/<run-tag>/cache/ (default: runs/)
--output PATH           Output parquet path
                        (default: <output-dir>/<run-tag>/features.parquet)
--workers INT           Concurrent raster sampling workers
```

---

### `lore/analysis.py`

Computes per-feature discriminability statistics and writes a human-readable
analysis report. Intended to be reviewed before training to identify features
that carry no signal and could be excluded.

**Per-feature statistics:**
- Kruskal-Wallis H-test and p-value. For large H statistics at high n,
  p-values underflow float64 precision; values are clamped to 5e-324 in
  the CSV and displayed as `p<2e-308` in the report.
- Mutual information with class label (bits)
- One-way ANOVA F-statistic

**Soil block aggregate:**
- MANOVA on the full 123-dimensional soil vector
- Summed MI across all soil features

**Signal ratings:**

| Rating | Criteria |
|---|---|
| STRONG | KW p < 0.001 and MI > 0.05 |
| MODERATE | KW p < 0.05 and MI > 0.01 |
| WEAK | KW p < 0.05 and MI <= 0.01 |
| FLAT | KW p >= 0.05 |

FLAT features can be safely excluded via `--exclude-features` in
`run_pipeline.py` or `lore/model.py`. WEAK features are low signal but not
noise; exclusion is optional.

```
--features PATH         Path to features.parquet
--output-dir PATH       Directory for analysis report and feature_stats.csv
--run-tag TAG           Run tag (used in report header)
--workers INT           Parallel workers for statistics computation
```

---

### `lore/model.py`

Trains a multi-input PyTorch neural network (LoreNet) with five parallel
encoder streams that are concatenated and decoded to a softmax output head.

```
numeric input      --> [Linear->ELU->Dropout] x depth --> enc_out
soil input         --> [Linear->ELU->Dropout] x depth --> enc_out
land cover input   --> [Linear->ELU->Dropout] x depth --> enc_out
date input         --> [Linear->Tanh->Dropout] x depth --> enc_out
name input         --> Embedding -> [Linear->Tanh->Dropout] x depth --> enc_out
                                                                  |
                                  Concatenate <-----------------+
                                        |
                             [Linear->ELU->Dropout] x decoder_depth
                                        |
                                  Linear -> softmax
```

Separate streams are used because feature blocks have very different
statistical properties. Shared early layers allow high-MI features to
dominate gradients; separate encoders let each block develop its own
representation before the merge.

**Name stream:** automatically disabled when only one source taxon is present.
A constant embedding contributes no gradient signal; disabling it reduces
parameter count and avoids noise. The disable state is recorded in the
checkpoint `architecture` block and respected at inference time.

**Class imbalance:** inverse-frequency weights applied in CrossEntropyLoss.
A warning is raised if the max/min class count ratio exceeds 100:1.

**Missing data:** class-conditional mean imputation at training time.
Per-feature means computed per class on the training split only, then
serialized to the checkpoint for use at inference time.

**Checkpoint format:** the checkpoint is fully self-describing for inference.
The `architecture` block records all LoreNet constructor arguments
(`n_numeric`, `n_soil`, `n_vocab`, `n_classes`, `use_name_stream`).
Use `build_model_from_checkpoint()` from `lore.model` to reconstruct.
A features parquet and checkpoint from different runs are not compatible;
always use the checkpoint produced from the same run.

**Dropout guidance:** the optimal dropout rate is inversely correlated with
dataset size. The default (0.1) is appropriate for large datasets (~60k+
training records). For smaller datasets (~5k records), 0.25 is recommended.
See `docs/hyperparameter_sweep.md` for full sweep results.

**Excluding taxa:** there is no `--exclude-classes` flag by design. To
exclude a taxon, filter the features parquet before training:

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('runs/peromyscus_split_2026/features.parquet')
df[df['suggested_names'] != 'Taxon name'].to_parquet(
    'runs/peromyscus_split_2026/features_filtered.parquet', index=False)
"
```

This is intentionally explicit. A checkpoint trained without a taxon cannot
predict it at inference time, and that should be an auditable decision.

```
--features PATH                 Path to features.parquet
--run-tag TAG                   Run identifier
--output-dir PATH               Base output directory. Cache and model written to
                                <output-dir>/<run-tag>/cache/ (default: runs/)
--exclude-features FEAT [...]   feat_* columns to exclude from training
--confidence-threshold FLOAT    Softmax threshold for ml_low_confidence flag
--dropout FLOAT                 Dropout rate (default: 0.1). Increase to 0.2-0.25
                                for smaller datasets (~5k records)
--device auto|cpu|cuda|mps      Training device (default: auto)
```

**Library usage:**
```python
from lore.model import train, build_model_from_checkpoint, load_checkpoint

results = train(
    features="runs/peromyscus_split_2026/features.parquet",
    run_tag="peromyscus_split_2026",
    output_dir="runs",
    exclude_features=["feat_lat", "feat_lon"],
)

# Reconstruct model from checkpoint for custom inference
import torch
ckpt  = load_checkpoint("runs/peromyscus_split_2026/cache/model/checkpoint.pt")
model = build_model_from_checkpoint(ckpt, device=torch.device("cpu"))
```

---

### `lore/predict.py`

Runs inference on ambiguous records and produces the final disambiguated CSV.
Geo-resolved records are passed through without ML inference (they constitute
training data; running ML on them would be circular).

**Disambiguation methods:**

| Method | Description |
|---|---|
| `geo` | Single-label, spatially confirmed by `lore/geo.py` |
| `ml` | Ambiguous, disambiguated by model above confidence threshold |
| `ml_low_confidence` | Ambiguous, model prediction below `--confidence-threshold` |
| `excluded` | Excessive coordinate uncertainty, or NaN in active features |

**Output columns** (all original Darwin Core columns preserved, plus):

| Column | Description |
|---|---|
| `final_taxon` | Primary output: resolved taxon name, or null if excluded |
| `disambiguation_method` | One of the four methods above |
| `suggested_names` | Original `geo.py` label, preserved unchanged |
| `ml_prediction` | Top model prediction (null for geo-resolved records) |
| `ml_confidence` | Softmax max probability (null for geo-resolved records) |
| `ml_candidate_match` | Bool: for parapatric records, was `ml_prediction` one of the known candidates? (null otherwise) |
| `feat_has_nodata` | True if any raster feature was NaN for this record |

```
--features PATH                 Path to features.parquet
--checkpoint PATH               Path to checkpoint.pt
--output PATH                   Output CSV path
--confidence-threshold FLOAT    Flag predictions below this as ml_low_confidence
--impute-inference              Apply serialized class-conditional means to NaN
                                features instead of excluding records
--device auto|cpu|cuda|mps
```

---

### `lore/visualize.py`

Generates a spatial PNG figure showing all occurrence records coloured by
resolved taxon, overlaid on MDD range polygons and a Natural Earth basemap.

**Point styling:**
- Geo-resolved: small filled circle, 40% alpha
- ML-resolved: larger filled circle, full alpha, white edge
- ML low-confidence: larger filled circle, full alpha, yellow edge
- Excluded: small grey x marker

**Colour palette:** <=7 taxa uses the Wong (2011) 8-colour colorblind-safe
palette. >7 taxa falls back to matplotlib `tab20` (perceptually distinct;
colorblind safety not guaranteed at >7 taxa, a documented limitation of
categorical palettes at large N).

**Projection:** azimuthal equidistant, centred on the median occurrence
coordinate. Extreme outliers trimmed via `--clip-quantile` before computing
the bounding box.

```
--disambiguated PATH        Path to disambiguated.csv
--ranges-file PATH          Path to MDD .gpkg range file
--dest-taxa TAXON [...]     Destination taxon names to plot ranges for
--output PATH               Output PNG path
--basemap-dir PATH          Directory containing Natural Earth subdirectories
--buffer-deg FLOAT          Degrees of padding around occurrence bbox (default: 1.0)
--clip-quantile FLOAT       Quantile trim for bbox computation (default: 0.0005).
                            Increase to trim more aggressively, decrease to
                            include more of the occurrence extent
--scale FLOAT               Figure size multiplier (default: 1.0, approx 10 in wide).
                            Increase (e.g. 2.0) if spatial detail is insufficient
--dpi INT                   Output resolution (default: 300)
```

**Library usage:**
```python
from lore.visualize import visualize

visualize(
    disambiguated = "runs/peromyscus_split_2026/disambiguated.csv",
    ranges_file   = "lore/data/ranges/MDD_Rodentia.gpkg",
    dest_taxa     = ["Peromyscus maniculatus", "Peromyscus sonoriensis", ...],
    output        = "runs/peromyscus_split_2026/figures/map.png",
    basemap_dir   = "lore/data/basemap",
)
```

---

### `run_pipeline.py`

End-to-end pipeline orchestration. Executes all steps in order; each step
checks whether its primary output already exists and skips if so. Pass
`--force` to re-run all steps regardless.

```
--run-tag TAG                   Required. Run identifier, used to derive all output paths.
--source-taxa TAXON [...]       Required. Pre-split taxon name(s) to filter occurrences.
--dest-taxa TAXON [...]         Required. Post-split destination taxon names.
--gbif-doi DOI                  GBIF occurrence DOI. Mutually exclusive with --occurrences-file.
--occurrences-file PATH         Custom Darwin Core CSV. Mutually exclusive with --gbif-doi.
--mdd-group GROUP               MDD taxonomic group (default: Rodentia).
--data-dir PATH                 Run output directory.
                                Defaults to runs/<run-tag> if not specified.
--ranges-dir PATH               Shared MDD range geopackage directory
                                (default: lore/data/ranges).
--raster-dir PATH               Directory containing global rasters
                                (default: lore/data/rasters).
--basemap-dir PATH              Directory containing Natural Earth vectors
                                (default: lore/data/basemap).
--exclude-features FEAT [...]   feat_* columns to exclude from model training.
--confidence-threshold FLOAT    Softmax threshold for ml_low_confidence flag.
--dropout FLOAT                 Dropout rate for model training (default: 0.1).
                                Increase to 0.2-0.25 for smaller datasets (~5k records).
--workers INT                   Parallel workers (default: 8).
--device auto|cpu|cuda|mps      Training device.
--force                         Re-run all steps even if outputs exist.
--skip-download                 Skip data acquisition entirely.
--skip-rasters                  Skip WorldClim, soil, and basemap downloads.
```

**Pipeline steps:**
1. Data acquisition (`scripts/download_data.py`)
2. Geospatial disambiguation (`lore/geo.py`)
3. Raster preprocessing (`scripts/preprocess_rasters.py`)
4. Feature extraction (`lore/features.py`)
5. Feature analysis (`lore/analysis.py`)
6. Model training (`lore/model.py`)
7. Inference and final output (`lore/predict.py`)
8. Figure generation (`lore/visualize.py`)

---

## Individual Step Usage

Each module can be run independently without `run_pipeline.py`:

```bash
# Data acquisition
python scripts/download_data.py \
    --gbif-doi 10.15468/dl.3cv9hy \
    --mdd-group Rodentia \
    --output-dir runs/peromyscus_split_2026 \
    --ranges-dir lore/data/ranges

# Geospatial disambiguation
python -m lore.geo \
    --occurrences runs/peromyscus_split_2026/occurrences.csv \
    --ranges-file lore/data/ranges/MDD_Rodentia.gpkg \
    --source-taxa "Peromyscus maniculatus" \
    --dest-taxa "Peromyscus maniculatus" "Peromyscus sonoriensis" \
                "Peromyscus gambelii" "Peromyscus keeni" \
                "Peromyscus labecula" "Peromyscus arcticus" \
    --run-tag peromyscus_split_2026 \
    --output-dir runs \
    --workers 8

# Raster preprocessing
python scripts/preprocess_rasters.py \
    --run-tag peromyscus_split_2026 \
    --occurrences runs/peromyscus_split_2026/geo_disambiguated.parquet \
    --raster-dir lore/data/rasters \
    --output-dir runs \
    --workers 4

# Feature extraction
python -m lore.features \
    --occurrences runs/peromyscus_split_2026/geo_disambiguated.parquet \
    --run-tag peromyscus_split_2026 \
    --output-dir runs \
    --workers 8

# Feature analysis
python -m lore.analysis \
    --features runs/peromyscus_split_2026/features.parquet \
    --output-dir runs/peromyscus_split_2026/analysis \
    --run-tag peromyscus_split_2026 \
    --workers 8

# Model training
python -m lore.model \
    --features runs/peromyscus_split_2026/features.parquet \
    --run-tag peromyscus_split_2026 \
    --output-dir runs

# Inference
python -m lore.predict \
    --features runs/peromyscus_split_2026/features.parquet \
    --checkpoint runs/peromyscus_split_2026/cache/model/checkpoint.pt \
    --output runs/peromyscus_split_2026/disambiguated.csv

# Figure generation
python -m lore.visualize \
    --disambiguated runs/peromyscus_split_2026/disambiguated.csv \
    --ranges-file lore/data/ranges/MDD_Rodentia.gpkg \
    --dest-taxa "Peromyscus maniculatus" "Peromyscus sonoriensis" \
                "Peromyscus gambelii" "Peromyscus keeni" \
                "Peromyscus labecula" "Peromyscus arcticus" \
    --output runs/peromyscus_split_2026/figures/map.png \
    --basemap-dir lore/data/basemap
```

---

## Known Limitations

- **Taxonomic scope:** range map support is currently limited to mammals via
  the MDD. Extension to other groups would require a compatible source of
  expert-drawn range polygons with bulk programmatic access.

- **Soil raster sampling speed:** sampling 123 rasters can take up to ~2 hours
  depending on storage I/O speed and the geographic extent of the destination
  taxa. One-time cost per run; output parquet is cached.

- **Raster preprocessing RAM:** ~44 GB peak during soil clipping
  in `preprocess_rasters.py`. One-time cost.

- **Coordinate uncertainty ceiling:** records with uncertainty > 128,576 m
  (largest US county radius) are flagged `excessive_uncertainty` and excluded
  from spatial disambiguation and ML inference. Configurable via
  `load_occurrences(uncertainty_ceil=...)`.

- **Taxon name encoder:** fit on the full dataset and embedded in the
  checkpoint. A features parquet and checkpoint from different runs are not
  compatible; always use the checkpoint produced from the same run.

- **nodata records:** occurrences at raster coverage edges (coastal, ocean)
  may have NaN features (`feat_has_nodata=True`). Excluded from ML inference
  by default; pass `--impute-inference` to attempt disambiguation using
  serialized class-conditional means.

- **Geography dependence:** lat/lon features carry strong signal that
  partially approximates the range shapefiles. Run with
  `--exclude-features feat_lat feat_lon` to assess discriminability
  independent of geography.

- **Name stream:** disabled automatically when only one source taxon is
  present in the occurrence dataset. A constant embedding contributes no
  gradient signal. The disable state is recorded in the checkpoint and
  applied transparently at inference time.

- **Infraspecific occurrence records:** source taxon matching captures records
  filed under infraspecific names (e.g. `"Microtus arvalis obscurus"`) when
  the parent binomial is specified in `--source-taxa`. Records filed under a
  distinct accepted name or synonym (e.g. `"Microtus levis"`) must be included
  explicitly in `--source-taxa` to be captured.

- **Colorblind palette scaling:** the Wong (2011) palette is colorblind-safe
  for <=7 taxa. Above this threshold LORE falls back to matplotlib `tab20`,
  which is perceptually distinct but not colorblind-safe. No universally
  safe categorical palette exists for >7 colors.

- **License:** LORE source code is MIT licensed. Output datasets inherit the
  licenses of their source data (GBIF CC-BY 4.0, MDD CC-BY 4.0, WorldClim
  CC-BY 4.0, Hengl soil data CC-BY 4.0, EarthEnv land cover CC-BY-NC 4.0).
  The EarthEnv land cover data is licensed for non-commercial use only.
  Users are responsible for compliance with data provider terms.

---

## Data Sources and Citations

**Example Peromyscus GBIF Dataset**
GBIF.org (25 March 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.3cv9hy

**Example Chaetodipus GBIF Dataset**
GBIF.org (28 March 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.cdmqxu

**Occurrence records (GBIF)**
GBIF.org. GBIF Occurrence Download.
https://www.gbif.org/occurrence/download

**Range maps (MDD)**
Marsh et al. (2022). Mammal Diversity Database (range maps).
*Journal of Mammalogy* 103(1):1-14.
Zenodo. https://doi.org/10.5281/zenodo.6644197

**Bioclimatic variables and elevation (WorldClim)**
Fick SE, Hijmans RJ (2017). WorldClim 2: new 1-km spatial resolution
climate surfaces for global land areas. *International Journal of
Climatology* 37(12):4302-4315. https://doi.org/10.1002/joc.5086

**Soil great group probability rasters**
Hengl T, Nauman T (2018). Predicted USDA soil great groups at 250 m
(probabilities). Zenodo. https://doi.org/10.5281/zenodo.3528062

**Land cover rasters (EarthEnv)**
Tuanmu MN, Jetz W (2014). A global 1-km consensus land-cover product for
biodiversity and ecosystem modeling. *Global Ecology and Biogeography*
23(9):1031-1045. https://www.earthenv.org/landcover

**Basemap vectors (Natural Earth)**
Natural Earth. Free vector and raster map data.
https://www.naturalearthdata.com
