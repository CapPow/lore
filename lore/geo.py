"""
lore/geo.py

Geospatial taxonomic disambiguation for GBIF occurrence records.

Given a set of destination taxon range maps and a set of source occurrence
records (Darwin Core format), assigns each occurrence a suggested taxon label
based on spatial intersection with the provided ranges. Records that fall
outside all ranges or in regions of parapatry are flagged for downstream
ML disambiguation.

Expected Darwin Core columns (required):
    gbifID, decimalLatitude, decimalLongitude,
    coordinateUncertaintyInMeters, verbatimScientificName, eventDate

All other columns are passed through unchanged.

Usage (library)
---------------
    from lore.geo import load_ranges, load_occurrences, disambiguate, describe_results

    ranges = load_ranges("lore/data/ranges/MDD_Rodentia.gpkg", taxa=[...])
    occ    = load_occurrences("runs/peromyscus_split_2026/occurrences.csv",
                              source_taxa=["Peromyscus maniculatus"])
    occ    = disambiguate(occ, ranges, processes=8)
    describe_results(occ, ranges)
    occ.to_parquet("runs/peromyscus_split_2026/geo_disambiguated.parquet", index=False)

Usage (CLI)
-----------
    python -m lore.geo \\
        --occurrences  runs/peromyscus_split_2026/occurrences.csv \\
        --ranges-file  lore/data/ranges/MDD_Rodentia.gpkg \\
        --source-taxa  "Peromyscus maniculatus" \\
        --dest-taxa    "Peromyscus maniculatus" "Peromyscus sonoriensis" \\
                       "Peromyscus gambelii" "Peromyscus keeni" \\
                       "Peromyscus labecula" "Peromyscus arcticus" \\
        --run-tag      peromyscus_split_2026 \\
        --output-dir   runs \\
        [--output      runs/peromyscus_split_2026/geo_disambiguated.parquet] \\
        [--name-col    sciname] \\
        [--workers     8] \\
        [--allopatry-threshold 0.1] \\
        [--strict-allopatry]
"""

import argparse
import gc
import warnings
import multiprocessing
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Largest US county area ≈ circle with radius 128,576 m
DEFAULT_UNCERTAINTY_CEIL  = 128_576   # metres
DEFAULT_UNCERTAINTY_FLOOR = 1         # metres

REQUIRED_OCC_COLS = [
    "gbifID",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "verbatimScientificName",
    "eventDate",
]

# Suggested-name sentinel values
LABEL_OUT_OF_RANGE     = "out_of_range"
LABEL_EXCESSIVE_UNCERT = "excessive_uncertainty"
LABEL_PARAPATRIC_SEP   = " | "   # delimiter when >1 range matched

PROJECT_ROOT   = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ranges(
    path: str | Path,
    taxa: list[str],
    name_col: str = "sciname",
) -> gpd.GeoDataFrame:
    """
    Load and filter destination taxon range polygons.

    Parameters
    ----------
    path : str or Path
        Path to any geopandas-readable vector file (.gpkg, .shp, .geojson …).
    taxa : list of str
        Taxon names to retain. Each entry may be a binomial species name
        (e.g. "Peromyscus maniculatus") or a genus name (e.g. "Peromyscus"),
        in which case all species in that genus are retained.
    name_col : str
        Column in the range file containing taxon names. Default: "sciname".

    Returns
    -------
    GeoDataFrame in EPSG:4326, filtered to requested taxa.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Range file not found: {path}")

    gdf = gpd.read_file(path)

    if name_col not in gdf.columns:
        raise ValueError(
            f"name_col '{name_col}' not found in range file. "
            f"Available columns: {gdf.columns.tolist()}"
        )

    taxa = [t.strip() for t in taxa]
    genera  = {t.lower() for t in taxa if len(t.split()) == 1}
    species = {t.lower() for t in taxa if len(t.split()) >= 2}

    col  = gdf[name_col].str.lower()
    mask = pd.Series(False, index=gdf.index)
    if genera:
        mask |= col.str.split().str[0].isin(genera)
    if species:
        mask |= col.isin(species)

    gdf = gdf[mask].copy()

    if len(gdf) == 0:
        raise ValueError(
            f"No ranges matched the requested taxa.\n"
            f"Requested: {taxa}\n"
            f"Sample values in '{name_col}': "
            f"{list(gpd.read_file(path)[name_col].head(10))}"
        )

    if gdf.crs is None:
        warnings.warn("Range file has no CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf.sindex  # build spatial index
    return gdf


def _validate_occurrences(
    occ: pd.DataFrame,
    source_taxa: list[str] | None = None,
) -> None:
    """
    Validate a Darwin Core occurrence DataFrame before processing.
    Raises ValueError for hard failures, warns for soft ones.
    Called from load_occurrences() for all input sources.
    """
    # ---- required columns ---------------------------------------------------
    missing = [c for c in REQUIRED_OCC_COLS if c not in occ.columns]
    if missing:
        # check for common casing variants to give actionable suggestions
        col_lower   = {c.lower(): c for c in occ.columns}
        suggestions = []
        for m in missing:
            if m.lower() in col_lower:
                suggestions.append(f"  '{col_lower[m.lower()]}' → rename to '{m}'")
        msg = (
            f"Occurrence file is missing required Darwin Core columns: {missing}\n"
            f"Required columns: {REQUIRED_OCC_COLS}"
        )
        if suggestions:
            msg += "\nPossible fixes (column rename required):\n" + "\n".join(suggestions)
        raise ValueError(msg)

    # ---- gbifID uniqueness --------------------------------------------------
    n_dupes = occ["gbifID"].duplicated().sum()
    if n_dupes:
        warnings.warn(
            f"{n_dupes} duplicate gbifID values found. "
            f"Records with duplicate IDs may produce unexpected join behaviour. "
            f"Consider deduplicating before running LORE."
        )

    # ---- coordinate numerics ------------------------------------------------
    lat = pd.to_numeric(occ["decimalLatitude"],  errors="coerce")
    lon = pd.to_numeric(occ["decimalLongitude"], errors="coerce")

    n_lat_bad = lat.isna().sum()
    n_lon_bad = lon.isna().sum()
    if n_lat_bad or n_lon_bad:
        raise ValueError(
            f"Non-numeric coordinate values found: "
            f"{n_lat_bad} in decimalLatitude, {n_lon_bad} in decimalLongitude. "
            f"Coordinates must be decimal degrees (e.g. 40.7128, -74.0060). "
            f"Degrees-minutes-seconds and projected coordinates are not accepted."
        )

    # ---- coordinate range ---------------------------------------------------
    lat_oor = ((lat < -90)  | (lat > 90)).sum()
    lon_oor = ((lon < -180) | (lon > 180)).sum()
    if lat_oor or lon_oor:
        raise ValueError(
            f"Coordinate values outside valid degree range: "
            f"{lat_oor} latitudes outside [-90, 90], "
            f"{lon_oor} longitudes outside [-180, 180]. "
            f"Coordinates must be decimal degrees in EPSG:4326. "
            f"If your data uses projected coordinates (metres), reproject first."
        )

    # ---- zero coordinates ---------------------------------------------------
    zero_mask  = (lat == 0) & (lon == 0)
    zero_ratio = zero_mask.sum() / max(len(occ), 1)
    if zero_ratio > 0.10:
        warnings.warn(
            f"{zero_mask.sum()} records ({zero_ratio:.1%}) have coordinates at "
            f"exactly (0, 0). These are likely default fill values and will be "
            f"dropped. If >10% of your data is affected, check your source file."
        )

    # ---- eventDate parse rate -----------------------------------------------
    dates           = pd.to_datetime(occ["eventDate"], errors="coerce")
    date_fail_ratio = dates.isna().sum() / max(len(occ), 1)
    if date_fail_ratio > 0.33:
        warnings.warn(
            f"eventDate could not be parsed for {date_fail_ratio:.1%} of records. "
            f"Expected ISO 8601 format (YYYY-MM-DD or YYYY). "
            f"Unparseable dates will produce null temporal features."
        )

    # ---- source taxa present ------------------------------------------------
    if source_taxa:
        source_taxa_clean = [t.strip() for t in source_taxa]
        genera  = {t.lower() for t in source_taxa_clean if len(t.split()) == 1}
        species = {t.lower() for t in source_taxa_clean if len(t.split()) >= 2}
        vsn  = occ["verbatimScientificName"].str.lower()
        mask = pd.Series(False, index=occ.index)
        if genera:
            mask |= vsn.str.split().str[0].isin(genera)
        if species:
            mask |= vsn.isin(species)
        if not mask.any():
            warnings.warn(
                f"None of the source_taxa {source_taxa_clean} were found in "
                f"verbatimScientificName. Check taxon name spelling and that "
                f"verbatimScientificName contains the pre-split source name "
                f"(not destination taxon names)."
            )

    # ---- empty after basic checks -------------------------------------------
    viable = occ.dropna(subset=["decimalLatitude", "decimalLongitude"])
    viable = viable[~((viable["decimalLatitude"] == 0) & (viable["decimalLongitude"] == 0))]
    if len(viable) == 0:
        raise ValueError(
            "No usable records remain after removing null and zero coordinates. "
            "Check that decimalLatitude and decimalLongitude are populated."
        )


def load_occurrences(
    path: str | Path | pd.DataFrame,
    source_taxa: list[str] | None = None,
    uncertainty_floor: float = DEFAULT_UNCERTAINTY_FLOOR,
    uncertainty_ceil: float  = DEFAULT_UNCERTAINTY_CEIL,
    uncertainty_fill: str | float = "median",
) -> gpd.GeoDataFrame:
    """
    Load, validate, and clean GBIF Darwin Core occurrence records.

    Parameters
    ----------
    path : str, Path, or DataFrame
        Path to a GBIF occurrence file (.csv or .txt/tsv), or an
        already-loaded DataFrame.
    source_taxa : list of str, optional
        If provided, filter occurrences to these taxon names before
        processing. Accepts genus or binomial strings, same as load_ranges().
    uncertainty_floor : float
        Minimum accepted coordinateUncertaintyInMeters. Values below this
        are raised to the floor. Default: 1 m.
    uncertainty_ceil : float
        Maximum accepted coordinateUncertaintyInMeters. Records exceeding
        this are flagged as LABEL_EXCESSIVE_UNCERT in suggested_names and
        excluded from spatial disambiguation. Default: 128,576 m.
    uncertainty_fill : str or float
        Value or method ("median" or "mean") used to impute null
        coordinateUncertaintyInMeters. Default: "median".

    Returns
    -------
    GeoDataFrame in EPSG:4326 with point geometry, ready for disambiguation.
    """
    # ---- load ---------------------------------------------------------------
    if isinstance(path, pd.DataFrame):
        occ = path.copy()
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Occurrence file not found: {path}")
        with open(path, "r") as _f:
            _first = _f.readline()
        sep = "\t" if "\t" in _first else ","
        occ = pd.read_csv(
            path,
            sep=sep,
            low_memory=False,
            on_bad_lines="warn",
        )

    # ---- validate -----------------------------------------------------------
    _validate_occurrences(occ, source_taxa=source_taxa)

    # ---- optional taxon filter ----------------------------------------------
    if source_taxa:
        source_taxa = [t.strip() for t in source_taxa]
        genera  = {t.lower() for t in source_taxa if len(t.split()) == 1}
        species = {t.lower() for t in source_taxa if len(t.split()) >= 2}
        mask = pd.Series(False, index=occ.index)
        if genera:
            mask |= occ["verbatimScientificName"].str.lower().str.split().str[0].isin(genera)
        if species:
            mask |= occ["verbatimScientificName"].str.lower().isin(species)
        occ = occ[mask].copy()

    # ---- drop missing coordinates -------------------------------------------
    orig_n = len(occ)
    occ    = occ.dropna(subset=["decimalLatitude", "decimalLongitude"])
    # some GBIF records encode missing coords as 0,0
    occ    = occ[~((occ["decimalLatitude"] == 0) & (occ["decimalLongitude"] == 0))]
    dropped = orig_n - len(occ)
    if dropped:
        warnings.warn(f"Dropped {dropped} records with missing/zero coordinates.")

    # ---- parse eventDate ----------------------------------------------------
    occ["eventDate"] = pd.to_datetime(occ["eventDate"], errors="coerce")

    # ---- uncertainty: floor -------------------------------------------------
    unc = occ["coordinateUncertaintyInMeters"].copy()
    unc = unc.clip(lower=uncertainty_floor)

    # ---- uncertainty: compute fill value ------------------------------------
    if isinstance(uncertainty_fill, str):
        uncertainty_fill = uncertainty_fill.lower()
        valid = unc[(unc <= uncertainty_ceil) & unc.notna()]
        if uncertainty_fill == "median":
            fill_val = valid.median()
        elif uncertainty_fill == "mean":
            fill_val = valid.mean()
        else:
            raise ValueError("uncertainty_fill must be 'median', 'mean', or a number.")
    else:
        fill_val = float(uncertainty_fill)

    null_ratio = unc.isna().sum() / len(unc)
    if null_ratio > 0.33:
        warnings.warn(
            f"High proportion ({null_ratio:.1%}) of null coordinateUncertaintyInMeters. "
            f"Imputing with {uncertainty_fill} = {fill_val:.1f} m."
        )
    unc = unc.fillna(fill_val)
    occ["coordinateUncertaintyInMeters"] = unc

    # ---- uncertainty: ceiling — flag excessive records ----------------------
    occ["suggested_names"] = ""
    excessive_mask = occ["coordinateUncertaintyInMeters"] > uncertainty_ceil
    occ.loc[excessive_mask, "suggested_names"] = LABEL_EXCESSIVE_UNCERT
    if excessive_mask.sum():
        warnings.warn(
            f"{excessive_mask.sum()} records flagged as '{LABEL_EXCESSIVE_UNCERT}' "
            f"(coordinateUncertaintyInMeters > {uncertainty_ceil} m)."
        )

    # ---- convert to GeoDataFrame --------------------------------------------
    occ = gpd.GeoDataFrame(
        occ,
        geometry=gpd.points_from_xy(occ["decimalLongitude"], occ["decimalLatitude"]),
        crs="EPSG:4326",
    )
    return occ


def disambiguate(
    occurrences: gpd.GeoDataFrame,
    ranges: gpd.GeoDataFrame,
    name_col: str = "sciname",
    processes: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Assign suggested taxon labels to occurrence records via spatial
    intersection with destination range polygons.

    Each occurrence's coordinate uncertainty is used to construct a circular
    polygon representing the collection region. That polygon is sampled as a
    MultiPoint (centroid + 128 perimeter points) and spatially joined to the
    range polygons. Records where all sample points fall within one range
    receive a single label. Records overlapping multiple ranges receive a
    pipe-delimited label. Records outside all ranges receive LABEL_OUT_OF_RANGE.
    Records already labelled LABEL_EXCESSIVE_UNCERT are passed through unchanged.

    Parameters
    ----------
    occurrences : GeoDataFrame
        Output of load_occurrences().
    ranges : GeoDataFrame
        Output of load_ranges().
    name_col : str
        Column in ranges containing taxon names. Default: "sciname".
    processes : int, optional
        Number of parallel processes. Default: cpu_count - 1, minimum 1.

    Returns
    -------
    GeoDataFrame with updated suggested_names column.
    """
    if processes is None:
        processes = max(1, multiprocessing.cpu_count() - 1)

    # build projected CRS centred on the range extent for accurate buffering
    bounds  = ranges.total_bounds  # (minx, miny, maxx, maxy)
    ref_lon = round(np.mean([bounds[0], bounds[2]]), 5)
    ref_lat = round(np.mean([bounds[1], bounds[3]]), 5)
    proj_crs = f"+proj=cea +lat_0={ref_lat} +lon_0={ref_lon} +units=m"

    # separate out pre-labelled excessive uncertainty records
    excessive_mask = occurrences["suggested_names"] == LABEL_EXCESSIVE_UNCERT
    excessive      = occurrences[excessive_mask].copy()
    to_process     = occurrences[~excessive_mask].copy()
    to_process     = to_process.drop(columns=["suggested_names"])

    print(
        f"Disambiguating {len(to_process):,} occurrences against "
        f"{len(ranges)} taxon ranges using {processes} process(es)."
    )

    indices = np.array_split(np.arange(len(to_process)), processes)
    chunks  = [to_process.iloc[idx] for idx in indices]
    del to_process

    results = Parallel(n_jobs=processes, prefer="processes", verbose=0)(
        delayed(_disambiguate_chunk)(chunk, ranges, name_col, proj_crs)
        for chunk in chunks
    )
    del chunks

    parts = [excessive] + results
    out   = pd.concat(parts, axis=0, ignore_index=True)
    del parts
    gc.collect()

    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


def allopatry_report(
    ranges: gpd.GeoDataFrame,
    name_col: str = "sciname",
    overlap_threshold: float = 0.1,
) -> tuple[bool, float, pd.DataFrame]:
    """
    Pairwise overlap analysis for destination ranges.

    Useful to verify that the requested taxa are sufficiently allopatric
    for geospatial disambiguation to be meaningful. Sympatric taxa will
    produce many parapatric records that must fall through to ML.

    Parameters
    ----------
    ranges : GeoDataFrame
        Output of load_ranges().
    name_col : str
        Column containing taxon names. Default: "sciname".
    overlap_threshold : float
        Maximum acceptable pairwise overlap ratio (0–1). Default: 0.1.

    Returns
    -------
    allopatric : bool
        True if all pairwise overlaps are below overlap_threshold.
    concept_wide_overlap : float
        Combined overlap area / total range area across all pairs.
    report : DataFrame
        Pairwise overlap ratios with columns:
        A_name, B_name, (A∩B)/A, (A∩B)/B.
    """
    bounds  = ranges.total_bounds
    ref_lon = round(np.mean([bounds[0], bounds[2]]), 5)
    ref_lat = round(np.mean([bounds[1], bounds[3]]), 5)
    proj_crs = f"+proj=cea +lat_0={ref_lat} +lon_0={ref_lon} +units=m"

    gdf      = ranges.to_crs(proj_crs)
    concepts = gdf[name_col].unique().tolist()

    rows = []
    for concept_a, concept_b in combinations(concepts, 2):
        geom_a = gdf[gdf[name_col] == concept_a].geometry.iloc[0]
        geom_b = gdf[gdf[name_col] == concept_b].geometry.iloc[0]
        if geom_a.intersects(geom_b):
            intersection = geom_a.intersection(geom_b)
            ratio_a = intersection.area / geom_a.area
            ratio_b = intersection.area / geom_b.area
        else:
            ratio_a = ratio_b = 0.0
        rows.append({
            "A_name":  concept_a,
            "B_name":  concept_b,
            "(A∩B)/A": ratio_a,
            "(A∩B)/B": ratio_b,
        })

    report = pd.DataFrame(rows)

    all_overlaps = [
        a.intersection(b)
        for a, b in combinations(gdf.geometry.tolist(), 2)
    ]
    total_area           = gdf.geometry.area.sum()
    overlap_area         = gpd.GeoSeries(all_overlaps, crs=proj_crs).area.sum()
    concept_wide_overlap = overlap_area / total_area if total_area > 0 else 0.0

    if report.empty:
        allopatric = True
    else:
        allopatric = bool(
            (report[["(A∩B)/A", "(A∩B)/B"]] < overlap_threshold).all().all()
        )

    return allopatric, concept_wide_overlap, report


def describe_results(
    occurrences: gpd.GeoDataFrame,
    ranges: gpd.GeoDataFrame,
    name_col: str = "sciname",
) -> None:
    """
    Print a summary of geospatial disambiguation results to stdout.

    Parameters
    ----------
    occurrences : GeoDataFrame
        Output of disambiguate().
    ranges : GeoDataFrame
        The range GeoDataFrame used for disambiguation.
    name_col : str
        Column in ranges containing taxon names. Default: "sciname".
    """
    taxa    = ranges[name_col].unique().tolist()
    n_total = len(occurrences)
    col     = occurrences["suggested_names"]

    n_single     = col.isin(taxa).sum()
    n_parapatric = col.str.contains(r"\|", na=False).sum()
    n_oor        = (col == LABEL_OUT_OF_RANGE).sum()
    n_excessive  = (col == LABEL_EXCESSIVE_UNCERT).sum()
    n_ambiguous  = n_parapatric + n_oor

    print(f"\n{'='*55}")
    print(f"  LORE geospatial disambiguation summary")
    print(f"{'='*55}")
    print(f"  Total occurrences:          {n_total:>10,}")
    print(f"  Single label assigned:      {n_single:>10,}  ({n_single/n_total:.1%})")
    print(f"  Parapatric (>1 label):      {n_parapatric:>10,}  ({n_parapatric/n_total:.1%})")
    print(f"  Out of range:               {n_oor:>10,}  ({n_oor/n_total:.1%})")
    print(f"  Excessive uncertainty:      {n_excessive:>10,}  ({n_excessive/n_total:.1%})")
    print(f"  Remaining ambiguous:        {n_ambiguous:>10,}  ({n_ambiguous/n_total:.1%})")
    print(f"{'='*55}")
    print(f"  Per-taxon counts:")
    for taxon in taxa:
        n = (col == taxon).sum()
        print(f"    {taxon:<40} {n:>8,}  ({n/n_total:.1%})")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_polygon(geom):
    """
    Convert a polygon to a MultiPoint of centroid + 128 perimeter points.
    Used to approximate polygon-in-polygon intersection via point-in-polygon,
    which is significantly faster at scale.
    """
    pts = list(geom.exterior.coords)   # 128 perimeter points (resolution=32 buffer)
    try:
        pts.append(tuple(geom.centroid.coords[0]))
    except (IndexError, Exception):
        pass
    return MultiPoint(pts)


def _disambiguate_chunk(
    chunk: gpd.GeoDataFrame,
    ranges: gpd.GeoDataFrame,
    name_col: str,
    proj_crs: str,
) -> gpd.GeoDataFrame:
    """
    Process a chunk of occurrences against range polygons.
    Runs in a worker process — must not reference any shared mutable state.
    """
    # project to equal-area CRS for accurate buffering
    chunk = chunk.to_crs(proj_crs)

    # buffer each point by its coordinate uncertainty (metres)
    chunk = chunk.copy()
    chunk["geometry"] = chunk.geometry.buffer(
        chunk["coordinateUncertaintyInMeters"],
        resolution=32,          # 4*32+1 = 129 points including centroid
    )

    # reproject back to WGS84 for spatial join against ranges
    chunk = chunk.to_crs("EPSG:4326")

    # sample each uncertainty polygon as a MultiPoint
    chunk["_pts"] = chunk.geometry.apply(_sample_polygon)
    chunk = chunk.set_geometry("_pts", crs="EPSG:4326")

    # explode MultiPoint → one row per point, preserving gbifID
    slim = chunk[["gbifID", "_pts"]].copy()
    slim = slim.explode(index_parts=False)

    # spatial join: which range does each point fall within?
    joined = gpd.sjoin(
        slim,
        ranges[[name_col, "geometry"]],
        how="left",
        predicate="within",
    )

    # many-to-one consolidation: collect all matched range names per gbifID
    joined["suggested_names"] = (
        joined.groupby("gbifID")[name_col]
        .transform(lambda x: LABEL_PARAPATRIC_SEP.join(sorted(x.dropna().unique())))
    )
    joined = (
        joined[["gbifID", "suggested_names"]]
        .drop_duplicates(subset="gbifID")
        .reset_index(drop=True)
    )
    joined["suggested_names"] = joined["suggested_names"].replace("", LABEL_OUT_OF_RANGE)

    # restore original geometry (points, not polygons or multipoints)
    chunk = chunk.set_geometry("geometry", crs="EPSG:4326")
    chunk = chunk.drop(columns=["_pts"])
    chunk = chunk.merge(joined, on="gbifID", how="left")
    chunk["suggested_names"] = chunk["suggested_names"].fillna(LABEL_OUT_OF_RANGE)

    del slim, joined
    gc.collect()
    return chunk


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Geospatial taxonomic disambiguation of GBIF occurrence records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--occurrences", type=Path, required=True,
        help="Path to GBIF occurrence CSV (Darwin Core format).",
    )
    p.add_argument(
        "--ranges-file", type=Path, required=True,
        help="Path to MDD range map geopackage (e.g. lore/data/ranges/MDD_Rodentia.gpkg).",
    )
    p.add_argument(
        "--source-taxa", nargs="+", required=True,
        help="Pre-split taxon name(s) to filter occurrences. "
             "Accepts genus or binomial. "
             "E.g. --source-taxa 'Peromyscus maniculatus'",
    )
    p.add_argument(
        "--dest-taxa", nargs="+", required=True,
        help="Post-split destination taxon names to load ranges for. "
             "E.g. --dest-taxa 'Peromyscus maniculatus' 'Peromyscus sonoriensis'",
    )
    p.add_argument(
        "--run-tag", default=None,
        help="Run identifier. Used to derive default output path as "
             "<output-dir>/<run-tag>/geo_disambiguated.parquet. "
             "Required unless --output is provided explicitly.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Base output directory. Output is written to "
             "<output-dir>/<run-tag>/geo_disambiguated.parquet "
             f"unless --output overrides. Default: {DEFAULT_OUTPUT_DIR}",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Explicit output path for the disambiguated parquet. "
             "Overrides the <output-dir>/<run-tag>/... default.",
    )
    p.add_argument(
        "--name-col", default="sciname",
        help="Column in the range file containing taxon names. Default: sciname. "
             "Override if your range file uses a different field name.",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel processes for disambiguation. "
             "Default: cpu_count - 1.",
    )
    p.add_argument(
        "--allopatry-threshold", type=float, default=0.1,
        help="Maximum acceptable pairwise range overlap ratio (0–1) for the "
             "allopatry sanity check. Pairs exceeding this trigger a warning. "
             "Default: 0.1.",
    )
    p.add_argument(
        "--strict-allopatry", action="store_true",
        help="Raise an exception if any pairwise range overlap exceeds "
             "--allopatry-threshold, rather than just warning.",
    )
    p.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: WARNING (progress printed via print()).",
    )
    return p


def main() -> None:
    import logging
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- resolve output path ------------------------------------------------
    if args.output is not None:
        out_path = args.output
    elif args.run_tag is not None:
        out_path = Path(args.output_dir) / args.run_tag / "geo_disambiguated.parquet"
    else:
        _build_parser().error(
            "One of --run-tag or --output is required to determine the output path."
        )

    print("=" * 55)
    print("  LORE — geospatial disambiguation")
    print("=" * 55)
    print(f"  Occurrences  : {args.occurrences}")
    print(f"  Ranges file  : {args.ranges_file}")
    print(f"  Source taxa  : {args.source_taxa}")
    print(f"  Dest taxa    : {args.dest_taxa}")
    print(f"  Name col     : {args.name_col}")
    print(f"  Workers      : {args.workers or 'auto (cpu_count - 1)'}")
    print(f"  Output       : {out_path}")
    print("=" * 55)

    # ---- load ---------------------------------------------------------------
    ranges = load_ranges(args.ranges_file, taxa=args.dest_taxa, name_col=args.name_col)
    occ    = load_occurrences(args.occurrences, source_taxa=args.source_taxa)

    # ---- allopatry sanity check ---------------------------------------------
    print("\nRunning allopatry check...")
    allopatric, cw_overlap, allo_report = allopatry_report(
        ranges,
        name_col=args.name_col,
        overlap_threshold=args.allopatry_threshold,
    )

    print(f"\n  Concept-wide overlap : {cw_overlap:.3f}")
    print(f"  Allopatric (all pairs < {args.allopatry_threshold}) : {allopatric}")
    if len(allo_report) > 0:
        print("\n  Pairwise overlaps:")
        for _, row in allo_report.iterrows():
            flag = (
                "  *** OVERLAP WARNING ***"
                if row["(A∩B)/A"] >= args.allopatry_threshold
                or row["(A∩B)/B"] >= args.allopatry_threshold
                else ""
            )
            print(
                f"    {row['A_name']:<35} × {row['B_name']:<35}"
                f"  (A∩B)/A={row['(A∩B)/A']:.3f}  (A∩B)/B={row['(A∩B)/B']:.3f}{flag}"
            )

    if not allopatric:
        msg = (
            f"One or more destination taxon pairs have range overlap exceeding "
            f"threshold={args.allopatry_threshold}. Parapatric records will be "
            f"numerous; ML disambiguation will carry higher load. "
            f"Review the pairwise overlap table above."
        )
        if args.strict_allopatry:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)

    # ---- disambiguate -------------------------------------------------------
    occ = disambiguate(occ, ranges, name_col=args.name_col, processes=args.workers)
    describe_results(occ, ranges, name_col=args.name_col)

    # ---- write output -------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    occ.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
