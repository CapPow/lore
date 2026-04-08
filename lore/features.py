"""
lore/features.py

Feature extraction for LORE (Latent Occurrence Resolution Engine).

Reads geo-disambiguated occurrence records and a preprocessed raster cache,
and appends ML-ready feature columns to each record.

Feature columns produced
------------------------
Numeric (continuous, float32):
    feat_lat            decimalLatitude,  min-max normalised over dataset
    feat_lon            decimalLongitude, min-max normalised over dataset
    feat_elevation      metres, sampled from elevation.tif
    feat_slope          degrees, sampled from slope.tif
    feat_bio1           WorldClim BIO1  (mean annual temperature, ×10 °C)
    feat_bio4           WorldClim BIO4  (temperature seasonality)
    feat_bio7           WorldClim BIO7  (Temperature Annual Range)
    feat_bio12          WorldClim BIO12 (annual precipitation, mm)
    feat_bio15          WorldClim BIO15 (precipitation seasonality)

Soil probability vector (123-dim, float32, min-max normalised per class):
    feat_soil_<class>   one column per entry in soil_class_names.json

Land cover probability vector (12-dim, float32, min-max normalised per class):
    feat_lc_<class>     one column per EarthEnv consensus land cover class.
                        Values represent percentage cover (0-100) before
                        normalisation. Classes: needleleaf_trees,
                        evergreen_broadleaf_trees, deciduous_broadleaf_trees,
                        mixed_other_trees, shrubs, herbaceous, cultivated,
                        flooded_vegetation, urban, snow_ice, barren, open_water.

Date (cyclical, float32):
    feat_sin_doy        sin(2π × day-of-year / 365)
    feat_cos_doy        cos(2π × day-of-year / 365)
                        Both are 0.0 when eventDate is missing.

Taxon name:
    feat_taxon_name             GBIF species binomial joined with
                                infraspecificEpithet when present
                                (e.g. "Microtus arvalis orcadensis").
                                Retained for human readability.
    feat_taxon_name_encoded     integer index into the encoder vocabulary
                                (int32). 0 is reserved for __unknown__ to
                                support inference on unseen names.

    The encoder is fit on the FULL dataset (all records) and written to
    <run_dir>/taxon_name_encoder.json as {"name": int, ...}.
    Splitting MUST be stratified on feat_taxon_name so that every class
    present at fit time appears in both train and test sets.

Quality flag:
    feat_has_nodata     True if any numeric feat_* column for this row
                        contains NaN.  feat_taxon_name* columns are
                        excluded from this check.
                        Records flagged here should be treated as
                        unreliable — filter or impute before training.

Sampling
--------
Scalar rasters (elevation, slope, bioclim):
    Bilinear interpolation via 2×2 windowed reads — one small window per
    point, no full-band read. RAM cost is O(n_points). Concurrent across
    rasters via ThreadPoolExecutor.

Soil probability rasters (123 classes):
    Nearest-pixel via rasterio's src.sample() generator, collected into
    a numpy array via np.fromiter (no Python-level per-point loop).
    src.sample() decompresses only the blocks containing requested
    points — peak RAM is ~4–5 MB for all concurrent workers combined
    regardless of raster size or worker count.

    Runtime on spinning HDD: ~30 minutes at --workers 8 due to random
    seek latency (135k seeks × 123 rasters). This is a hardware limitation.
    On NVMe expect ~2–3 min. This is a one-time cost per run; the output
    parquet is cached.

Land cover rasters (12 classes):
    Bilinear interpolation via the same scalar sampling path as WorldClim.
    Sampled at unique coordinates only; results joined back to all records.

Normalisation
-------------
lat/lon, elevation, slope, bio1–15:
    min-max over the dataset being processed (not global).
soil:
    global min-max from soil_stats.json (written by preprocess_rasters.py).
    Using stored global stats keeps soil vectors comparable across runs.

Usage (library)
---------------
    from lore.features import extract_features

    df = extract_features(
        occurrences="runs/peromyscus_split_2026/geo_disambiguated.parquet",
        run_tag="peromyscus_split_2026",
        output_dir="runs",
        workers=8,
    )
    # features.parquet is written automatically to runs/peromyscus_split_2026/
    # Pass output= to override the destination path.

Usage (CLI)
-----------
    python -m lore.features \\
        --occurrences runs/peromyscus_split_2026/geo_disambiguated.parquet \\
        --run-tag     peromyscus_split_2026 \\
        --output-dir  runs \\
        [--output     runs/peromyscus_split_2026/features.parquet] \\
        [--workers    8]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT      = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"
DEFAULT_WORKERS   = 8

BIOCLIM_BANDS = [1, 4, 7, 12, 15]

ENCODER_FILENAME = "taxon_name_encoder.json"
UNKNOWN_TOKEN    = "__unknown__"

# Maximum date range span to attempt midpoint encoding.
# Ranges exceeding this convey no meaningful seasonal signal and are
# treated as missing. 60 days ~ within a single seasonal transition.
MAX_DATE_RANGE_DAYS = 60

# Coordinate rounding precision for raster sampling deduplication.
# 3 decimal places ~ 111m, slightly finer than the 250m soil raster resolution.
# Points rounding to the same value are guaranteed to sample the same soil cell,
# avoiding redundant I/O on systematic trap/survey datasets.
SAMPLE_COORD_DECIMALS = 3

# ---------------------------------------------------------------------------
# Scalar raster sampling — bilinear, 2×2 windowed
# ---------------------------------------------------------------------------

def _sample_scalar_bilinear(
    path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Sample a single-band raster at N (lon, lat) points using bilinear
    interpolation with 2×2 windowed reads. No full-band read.
    Points outside extent or adjacent to nodata → NaN.
    """
    out = np.full(len(lons), np.nan, dtype=np.float32)

    with rasterio.open(str(path)) as src:
        nodata    = src.nodata
        transform = src.transform
        height    = src.height
        width     = src.width

        cols_f = (lons - transform.c) / transform.a
        rows_f = (lats - transform.f) / transform.e

        for i in range(len(lons)):
            c0 = int(math.floor(cols_f[i]))
            r0 = int(math.floor(rows_f[i]))
            c1, r1 = c0 + 1, r0 + 1

            if c0 < 0 or r0 < 0 or c1 >= width or r1 >= height:
                continue

            window = rasterio.windows.Window(c0, r0, 2, 2)
            patch  = src.read(1, window=window).astype(np.float32)

            if nodata is not None:
                patch[patch == nodata] = np.nan

            q00, q01 = patch[0, 0], patch[0, 1]
            q10, q11 = patch[1, 0], patch[1, 1]

            if np.isnan(q00) or np.isnan(q01) or np.isnan(q10) or np.isnan(q11):
                continue

            dc = cols_f[i] - c0
            dr = rows_f[i] - r0
            out[i] = (
                q00 * (1 - dc) * (1 - dr)
                + q01 * dc      * (1 - dr)
                + q10 * (1 - dc) * dr
                + q11 * dc      * dr
            )

    return out


def _sample_scalar_rasters_parallel(
    paths: Sequence[Path],
    lons: np.ndarray,
    lats: np.ndarray,
    workers: int,
) -> dict[Path, np.ndarray]:
    """Sample scalar rasters concurrently. Returns {path: float32 array}."""
    results: dict[Path, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {
            pool.submit(_sample_scalar_bilinear, p, lons, lats): p
            for p in paths
        }
        for future in tqdm(
            as_completed(future_to_path),
            total=len(future_to_path),
            desc="scalar rasters",
            unit="raster",
        ):
            path = future_to_path[future]
            try:
                results[path] = future.result()
            except Exception as exc:
                logger.error("Error sampling %s: %s", path.name, exc)
                results[path] = np.full(len(lons), np.nan, dtype=np.float32)
    return results


# ---------------------------------------------------------------------------
# Soil raster sampling — nearest-pixel, src.sample(), concurrent
# ---------------------------------------------------------------------------

def _sample_soil_one(
    path: Path,
    name: str,
    coords: list[tuple[float, float]],
    n: int,
) -> tuple[str, np.ndarray]:
    """
    Sample one soil probability raster using src.sample() + np.fromiter.
    Peak RAM per call: ~0.5 MB (output array only). Safe at any concurrency.
    """
    with rasterio.open(str(path)) as src:
        nodata = src.nodata
        arr = np.fromiter(
            (float(v[0]) for v in src.sample(coords)),
            dtype=np.float32,
            count=n,
        )
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return name, arr


def _sample_soil_rasters_concurrent(
    paths: Sequence[Path],
    class_names: Sequence[str],
    lons: np.ndarray,
    lats: np.ndarray,
    workers: int,
) -> dict[str, np.ndarray]:
    """
    Sample all soil rasters concurrently.
    Peak RAM ~4–5 MB total regardless of worker count.
    Runtime is I/O bound — see module docstring for HDD vs NVMe estimates.
    Returns {class_name: float32 array of length N}.
    """
    n      = len(lons)
    coords = list(zip(lons.tolist(), lats.tolist()))

    results: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_name = {
            pool.submit(_sample_soil_one, path, name, coords, n): name
            for path, name in zip(paths, class_names)
        }
        for future in tqdm(
            as_completed(future_to_name),
            total=len(future_to_name),
            desc="soil rasters",
            unit="raster",
        ):
            name = future_to_name[future]
            try:
                _, arr = future.result()
                results[name] = arr
            except Exception as exc:
                logger.error("Error sampling soil raster %s: %s", name, exc)
                results[name] = np.full(n, np.nan, dtype=np.float32)

    return results


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _minmax_norm(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Min-max normalise to [0, 1]. Zero-range → 0.0. NaN propagated."""
    rng = vmax - vmin
    if rng == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    normed = ((arr - vmin) / rng).astype(np.float32)
    normed[np.isnan(arr)] = np.nan
    return normed


def _fit_minmax(arr: np.ndarray) -> tuple[float, float]:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    return float(valid.min()), float(valid.max())


# ---------------------------------------------------------------------------
# Taxon name helpers
# ---------------------------------------------------------------------------

def _build_taxon_name(species: str | None, infraspecific: str | None) -> str:
    """
    Build a clean taxon name for encoding from GBIF species and
    infraspecificEpithet columns.

    Combines species binomial with infraspecific epithet when present,
    giving the model signal about pre-split subspecific determinations
    which frequently correspond to post-split destination taxa.

    Falls back to empty string for null/missing input.
    """
    if not isinstance(species, str) or not species.strip():
        return ""
    name = species.strip()
    if isinstance(infraspecific, str) and infraspecific.strip():
        name = f"{name} {infraspecific.strip()}"
    return name.strip()


def fit_name_encoder(cleaned_names: pd.Series) -> dict[str, int]:
    """
    Build a {name: int} encoder. Vocabulary sorted for determinism.
    0 reserved for UNKNOWN_TOKEN; classes start at 1.
    """
    vocab = sorted(n for n in cleaned_names.dropna().unique() if n)
    encoder: dict[str, int] = {UNKNOWN_TOKEN: 0}
    for i, name in enumerate(vocab, start=1):
        encoder[name] = i
    return encoder


def apply_name_encoder(
    cleaned_names: pd.Series,
    encoder: dict[str, int],
) -> np.ndarray:
    """Encode cleaned names to int32. Absent names -> 0 (UNKNOWN_TOKEN)."""
    return cleaned_names.map(lambda n: encoder.get(n, 0)).to_numpy(dtype=np.int32)

# ---------------------------------------------------------------------------
# Date encoding
# ---------------------------------------------------------------------------
def _cyclical_doy(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Encode eventDate as sin/cos day-of-year. Missing dates -> (0.0, 0.0).

    Handles common GBIF eventDate formats beyond ISO 8601:
        - Date ranges (e.g. "1965-09-15/1965-09-30"): midpoint used if span
          is within MAX_DATE_RANGE_DAYS; otherwise treated as missing.
        - Year-month only (e.g. "1900-04"): day imputed to 15.
        - Year only: no seasonal information; set to 0.0 (same as missing).
    """
    cleaned = dates.astype(str).copy()

    # handle date ranges: "date1/date2"
    range_mask = cleaned.str.contains("/", na=False)
    if range_mask.any():
        parts = cleaned[range_mask].str.split("/")
        d1 = pd.to_datetime(parts.str[0], errors="coerce")
        d2 = pd.to_datetime(parts.str[1], errors="coerce")
        span = (d2 - d1).dt.days
        midpoint = d1 + pd.to_timedelta(span // 2, unit="D")
        midpoint[span > MAX_DATE_RANGE_DAYS] = pd.NaT
        cleaned[range_mask] = midpoint.dt.strftime("%Y-%m-%d").fillna("NaT")

    # try standard parse (handles full ISO dates, datetime with time component)
    parsed = pd.to_datetime(cleaned, errors="coerce")

    # year-month only: "YYYY-MM" — month is known, impute day to 15
    still_null = parsed.isna()
    if still_null.any():
        ym = pd.to_datetime(
            cleaned[still_null] + "-15", format="%Y-%m-%d", errors="coerce"
        )
        parsed = parsed.copy()
        parsed[still_null] = ym

    # year-only and truly missing remain NaT — no seasonal signal, leave as 0.0

    # .copy() required — pandas may return a read-only array from parquet
    doy  = parsed.dt.dayofyear.to_numpy(dtype=float).copy()
    mask = np.isnan(doy)
    doy[mask] = 0.0
    angle   = 2.0 * np.pi * doy / 365.0
    sin_doy = np.sin(angle).astype(np.float32)
    cos_doy = np.cos(angle).astype(np.float32)
    sin_doy[mask] = 0.0
    cos_doy[mask] = 0.0
    n_missing = int(mask.sum())
    if n_missing:
        logger.warning(
            "%d records have missing or unresolvable eventDate; "
            "date features set to 0.",
            n_missing,
        )
    return sin_doy, cos_doy


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_features(
    occurrences: str | Path,
    run_tag: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    """
    Extract ML features for all records in *occurrences*.

    Parameters
    ----------
    occurrences : path to geo_disambiguated.parquet
    run_tag     : run identifier (e.g. 'peromyscus_split_2026')
    output_dir  : base output directory; cache is read from
                  <output_dir>/<run_tag>/cache/
    workers     : parallel threads for raster sampling.
                  Soil RAM is ~4 MB total regardless of worker count.
                  Increase freely on fast storage to reduce wall time.

    Returns
    -------
    pandas.DataFrame — all original columns plus feat_* columns.

    Side effects
    ------------
    Writes <output_dir>/<run_tag>/cache/taxon_name_encoder.json.
    """
    occurrences = Path(occurrences)
    run_dir     = Path(output_dir) / run_tag / "cache"

    # ---- validate cache ----------------------------------------------------
    if not (run_dir / "bbox.json").exists():
        raise FileNotFoundError(
            f"Cache directory not found or incomplete: {run_dir}\n"
            f"Run scripts/preprocess_rasters.py --run-tag {run_tag} first."
        )
    for fname in ("soil_class_names.json", "soil_stats.json"):
        if not (run_dir / fname).exists():
            raise FileNotFoundError(f"{fname} missing in {run_dir}.")

    soil_class_names: list[str] = json.loads(
        (run_dir / "soil_class_names.json").read_text()
    )
    soil_stats: dict[str, dict] = json.loads(
        (run_dir / "soil_stats.json").read_text()
    )

    # land cover class names — optional; warn if missing rather than raising
    lc_names_path = run_dir / "landcover_class_names.json"
    if lc_names_path.exists():
        lc_class_names: list[str] = json.loads(lc_names_path.read_text())
    else:
        warnings.warn(
            f"landcover_class_names.json not found in {run_dir}. "
            f"Land cover features will be omitted. "
            f"Re-run preprocess_rasters.py to generate land cover cache."
        )
        lc_class_names = []

    # ---- load occurrences --------------------------------------------------
    logger.info("Loading occurrences from %s", occurrences)
    df = pd.read_parquet(occurrences)
    n  = len(df)
    logger.info("  %d records loaded.", n)

    lons = df["decimalLongitude"].to_numpy(dtype=np.float64)
    lats = df["decimalLatitude"].to_numpy(dtype=np.float64)

    # ---- deduplicate coordinates for raster sampling -----------------------
    # Many systematic survey datasets contain repeated trap coordinates.
    # Sampling unique locations only and joining back avoids redundant I/O,
    # particularly for the slow soil raster sampling step.
    coord_df = pd.DataFrame({"lon": lons, "lat": lats})
    coord_df["lon_r"] = coord_df["lon"].round(SAMPLE_COORD_DECIMALS)
    coord_df["lat_r"] = coord_df["lat"].round(SAMPLE_COORD_DECIMALS)
    unique_coords = coord_df.drop_duplicates(subset=["lon_r", "lat_r"])
    n_unique = len(unique_coords)
    if n_unique < n:
        logger.info(
            "  %d unique sampling locations from %d records (%.1f%% reduction).",
            n_unique, n, 100 * (1 - n_unique / n),
        )

    u_lons = unique_coords["lon"].to_numpy(dtype=np.float64)
    u_lats = unique_coords["lat"].to_numpy(dtype=np.float64)

    # ---- build raster path lists -------------------------------------------
    elev_path  = run_dir / "elevation.tif"
    slope_path = run_dir / "slope.tif"
    bio_paths  = {b: run_dir / f"wc2.1_30s_bio_{b}.tif" for b in BIOCLIM_BANDS}

    scalar_paths: list[Path] = []
    for p, label in [
        (elev_path,  "elevation.tif"),
        (slope_path, "slope.tif"),
        *[(bio_paths[b], f"wc2.1_30s_bio_{b}.tif") for b in BIOCLIM_BANDS],
    ]:
        if p.exists():
            scalar_paths.append(p)
        else:
            warnings.warn(
                f"{label} not found in {run_dir}; corresponding features will be NaN."
            )

    soil_paths = [run_dir / "soil" / f"{name}.tif" for name in soil_class_names]
    missing_soil = [p for p in soil_paths if not p.exists()]
    if missing_soil:
        raise FileNotFoundError(
            f"{len(missing_soil)} soil tifs missing from {run_dir / 'soil'}. "
            f"Re-run preprocess_rasters.py --force."
        )

    lc_paths = [run_dir / "landcover" / f"{name}.tif" for name in lc_class_names]
    missing_lc = [p for p in lc_paths if not p.exists()]
    if missing_lc and lc_class_names:
        warnings.warn(
            f"{len(missing_lc)} land cover tifs missing from {run_dir / 'landcover'}. "
            f"Re-run preprocess_rasters.py to regenerate."
        )
        lc_paths = [p for p in lc_paths if p.exists()]
        lc_class_names = [n for n, p in zip(lc_class_names, lc_paths)]

    # ---- sample rasters (on unique coordinates only) -----------------------
    logger.info(
        "Sampling %d scalar rasters with bilinear interpolation (%d workers)...",
        len(scalar_paths), workers,
    )
    scalar_results = _sample_scalar_rasters_parallel(
        scalar_paths, u_lons, u_lats, workers
    )

    logger.info(
        "Sampling %d soil probability rasters with nearest-pixel (%d workers)...",
        len(soil_paths), workers,
    )
    soil_results = _sample_soil_rasters_concurrent(
        soil_paths, soil_class_names, u_lons, u_lats, workers
    )
    
    logger.info(
        "Sampling %d land cover rasters with bilinear interpolation (%d workers)...",
        len(lc_paths), workers,
    )
    lc_results = _sample_scalar_rasters_parallel(
        lc_paths, u_lons, u_lats, workers
    ) if lc_paths else {}

    # ---- join raster results back to full record set -----------------------
    join_cols = {}
    for path, arr in scalar_results.items():
        join_cols[str(path)] = arr
    for name, arr in soil_results.items():
        join_cols[f"_soil_{name}"] = arr
    for name, path in zip(lc_class_names, lc_paths):
        join_cols[f"_lc_{name}"] = lc_results.get(path, np.full(n_unique, np.nan, dtype=np.float32))

    unique_coords = unique_coords.reset_index(drop=True)
    unique_coords = pd.concat(
        [unique_coords,
        pd.DataFrame(join_cols, index=unique_coords.index)],  # now both 0..n_unique
        axis=1,
    )

    coord_df = coord_df.merge(
        unique_coords.drop(columns=["lon", "lat"]),
        on=["lon_r", "lat_r"],
        how="left",
    )

    # rebuild scalar_results and soil_results indexed to full n records
    scalar_results = {
        path: coord_df[str(path)].to_numpy(dtype=np.float32)
        for path in scalar_paths
    }
    soil_results = {
        name: coord_df[f"_soil_{name}"].to_numpy(dtype=np.float32)
        for name in soil_class_names
    }
    
    lc_results = {
        name: coord_df[f"_lc_{name}"].to_numpy(dtype=np.float32)
        for name in lc_class_names
    }

    # ---- assemble feature columns ------------------------------------------
    # Accumulate all columns in a dict, then pd.concat once — avoids
    # PerformanceWarning from repeated single-column DataFrame insertion.
    _nan = np.full(n, np.nan, dtype=np.float32)
    feat_cols: dict[str, np.ndarray] = {}

    # lat / lon
    lat_arr = lats.astype(np.float32)
    lon_arr = lons.astype(np.float32)
    feat_cols["feat_lat"] = _minmax_norm(lat_arr, *_fit_minmax(lat_arr))
    feat_cols["feat_lon"] = _minmax_norm(lon_arr, *_fit_minmax(lon_arr))

    # elevation & slope
    elev_raw  = scalar_results.get(elev_path,  _nan)
    slope_raw = scalar_results.get(slope_path, _nan)
    feat_cols["feat_elevation"] = _minmax_norm(elev_raw,  *_fit_minmax(elev_raw))
    feat_cols["feat_slope"]     = _minmax_norm(slope_raw, *_fit_minmax(slope_raw))

    # bioclim
    for band in BIOCLIM_BANDS:
        raw = scalar_results.get(bio_paths[band], _nan)
        feat_cols[f"feat_bio{band}"] = _minmax_norm(raw, *_fit_minmax(raw))

    # soil (global min-max from soil_stats.json)
    for name in soil_class_names:
        raw  = soil_results[name]
        stat = soil_stats.get(name, {})
        feat_cols[f"feat_soil_{name}"] = _minmax_norm(
            raw, float(stat.get("min", 0.0)), float(stat.get("max", 1.0))
        )

    # land cover (run-local min-max, same as bioclim)
    for name in lc_class_names:
        raw = lc_results.get(name, _nan)
        feat_cols[f"feat_lc_{name}"] = _minmax_norm(raw, *_fit_minmax(raw))

    # date
    sin_doy, cos_doy = _cyclical_doy(df["eventDate"])
    feat_cols["feat_sin_doy"] = sin_doy
    feat_cols["feat_cos_doy"] = cos_doy

    # taxon name — species + infraspecificEpithet when present
    species_col = df["species"] if "species" in df.columns else pd.Series(
        [""] * n, index=df.index
    )
    infra_col = df["infraspecificEpithet"] if "infraspecificEpithet" in df.columns else pd.Series(
        [None] * n, index=df.index
    )
    taxon_names = pd.Series(
        [_build_taxon_name(s, i) for s, i in zip(species_col, infra_col)],
        index=df.index,
    )
    encoder = fit_name_encoder(taxon_names)
    feat_cols["feat_taxon_name"]         = taxon_names.to_numpy(dtype=object)
    feat_cols["feat_taxon_name_encoded"] = apply_name_encoder(taxon_names, encoder)

    # nodata flag — numeric features only
    numeric_keys = [
        k for k in feat_cols
        if k not in {"feat_taxon_name", "feat_taxon_name_encoded"}
    ]
    numeric_matrix = np.stack([feat_cols[k] for k in numeric_keys], axis=1)
    feat_cols["feat_has_nodata"] = np.isnan(numeric_matrix).any(axis=1)

    # single concat
    df = pd.concat(
        [df, pd.DataFrame(feat_cols, index=df.index)],
        axis=1,
    )

    # ---- serialize encoder -------------------------------------------------
    encoder_path = run_dir / ENCODER_FILENAME
    encoder_path.write_text(json.dumps(encoder, indent=2, ensure_ascii=False))
    logger.info(
        "Name encoder (%d classes) written to %s", len(encoder) - 1, encoder_path
    )

    n_nodata = int(df["feat_has_nodata"].sum())
    if n_nodata:
        warnings.warn(
            f"{n_nodata} records ({100 * n_nodata / n:.1f}%) have at least one NaN feature "
            f"(feat_has_nodata=True). These are unreliable — filter or impute before training.",
            stacklevel=2,
        )

    logger.info(
        "Feature extraction complete. %d records, %d numeric feat columns, %d name classes.",
        n, len(numeric_keys), len(encoder) - 1,
    )
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract ML features from a LORE raster cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--occurrences", type=Path, required=True,
                   help="Path to geo_disambiguated.parquet.")
    p.add_argument("--run-tag", required=True,
                   help="Run identifier (e.g. 'peromyscus_split_2026'). "
                        "Cache is read from <output-dir>/<run-tag>/cache/.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Base output directory. Cache is read from "
                        "<output-dir>/<run-tag>/cache/. "
                        f"Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--output", type=Path, default=None,
                   help="Output parquet path. "
                        "Default: <output-dir>/<run-tag>/features.parquet.")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=(
                       "Parallel threads for raster sampling. Soil RAM is ~4 MB "
                       "total regardless of worker count. Increase on fast storage."
                   ))
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run_dir = Path(args.output_dir) / args.run_tag / "cache"
    output  = args.output or (Path(args.output_dir) / args.run_tag / "features.parquet")

    print("=" * 55)
    print("  LORE — feature extraction")
    print("=" * 55)
    print(f"  Occurrences : {args.occurrences}")
    print(f"  Run tag     : {args.run_tag}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Cache dir   : {run_dir}")
    print(f"  Workers     : {args.workers}")
    print(f"  Output      : {output}")
    print("=" * 55)

    df = extract_features(
        occurrences=args.occurrences,
        run_tag=args.run_tag,
        output_dir=args.output_dir,
        workers=args.workers,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    numeric_feat_cols = [
        c for c in df.columns
        if c.startswith("feat_")
        and c not in {"feat_taxon_name", "feat_taxon_name_encoded", "feat_has_nodata"}
    ]
    n_nodata  = int(df["feat_has_nodata"].sum())
    n_classes = int(df["feat_taxon_name_encoded"].max())

    print()
    print(f"  Records          : {len(df):,}")
    print(f"  Numeric features : {len(numeric_feat_cols)}")
    print(f"  Name classes     : {n_classes}")
    print(f"  Nodata records   : {n_nodata:,}  ({100 * n_nodata / len(df):.1f}%)")
    print(f"  Encoder saved    : {run_dir / ENCODER_FILENAME}")
    print(f"  Output saved     : {output}")
    print("=" * 55)


if __name__ == "__main__":
    main()
