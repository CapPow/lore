"""
scripts/download_data.py

One-time acquisition of all data required for a LORE disambiguation run.
Downloads are skipped if output files already exist. Re-download by deleting
the relevant file and re-running.

Downloads
---------
Run-specific (one of --gbif-doi or --occurrences-file required):
    GBIF occurrence records (SIMPLE_CSV, DWC fields, tab-delimited)
        Accepts DOI, GBIF download URL, or raw download key.

    OR custom occurrence CSV (Darwin Core compatible)
        Must contain required columns:
            gbifID, decimalLatitude, decimalLongitude,
            coordinateUncertaintyInMeters, verbatimScientificName, eventDate
        Coordinates must be decimal degrees (EPSG:4326).
        Full validation occurs in lore/geo.py:load_occurrences() at runtime.

    MDD taxonomic group range maps (geopackage, Deflate64 zip)
        Source: Zenodo DOI 10.5281/zenodo.<record>
        Default: MDD v1.2, Rodentia (Zenodo record 6644198)
        Citation: Marsh et al. (2022) Journal of Mammalogy 103:1-14

Global (shared across runs):
    WorldClim v2.1 bioclimatic variables + elevation (30 arc-second, ~1 km)
        Source: https://worldclim.org/data/worldclim21.html
        Citation: Fick & Hijmans (2017) Int. J. Climatology 37:4302-4315

    USDA Soil Great Group probability rasters (250m, 123 classes)
        Source: Zenodo DOI 10.5281/zenodo.3528062
        Citation: Hengl & Nauman (2018) Zenodo

    Natural Earth 1:10m cultural + physical vectors (basemap)
        Source: https://www.naturalearthdata.com

Default output layout
---------------------
    <output-dir>/occurrences.csv

    <ranges-dir>/MDD_<Group>.gpkg

    <raster-dir>/worldclim/wc2.1_30s_elev.tif
    <raster-dir>/worldclim/wc2.1_30s_bio_<N>.tif
    <raster-dir>/soil/sol_grtgroup_usda.soiltax.*_p_250m*.tif
    <raster-dir>/soil/sol_grtgroup_usda.soiltax_c_250m*.tif

    <basemap-dir>/ne_10m_admin_0_countries/
    <basemap-dir>/ne_10m_admin_1_states_provinces/
    <basemap-dir>/ne_10m_land/

Usage
-----
    # List available MDD taxonomic groups for a given Zenodo record:
    python scripts/download_data.py --list-mdd-groups

    # Full acquisition via GBIF DOI:
    python scripts/download_data.py \
        --gbif-doi 10.15468/dl.3cv9hy \
        --output-dir runs/peromyscus_split_2026

    # Full acquisition via custom occurrence file:
    python scripts/download_data.py \
        --occurrences-file path/to/occurrences.csv \
        --output-dir runs/peromyscus_split_2026

    # Different MDD group:
    python scripts/download_data.py \
        --gbif-doi 10.15468/dl.xxxx \
        --output-dir runs/my_run \
        --mdd-group Carnivora

    # Global data only (no occurrence download, --output-dir not required):
    python scripts/download_data.py --skip-occurrences --skip-ranges

    # Skip large global downloads if already present:
    python scripts/download_data.py \
        --gbif-doi 10.15468/dl.xxxx \
        --output-dir runs/my_run \
        --skip-worldclim --skip-soil --skip-basemap
"""

import argparse
import io
import shutil
import sys
import zipfile
import zipfile_inflate64  # noqa: F401 — patches zipfile to support Deflate64 (type 9)
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE         = Path(__file__).parent
PROJECT_ROOT = HERE.parent

DEFAULT_RANGES_DIR  = PROJECT_ROOT / "lore"  / "data" / "ranges"
DEFAULT_RASTER_DIR  = PROJECT_ROOT / "lore"  / "data" / "rasters"
DEFAULT_BASEMAP_DIR = PROJECT_ROOT / "lore"  / "data" / "basemap"

# MDD range maps - Zenodo
# 6644197 is the concept DOI record (version-agnostic); resolves to latest.
# 6644198 was the pinned v1.2 record.
MDD_ZENODO_RECORD   = "6644197"
MDD_ZENODO_BASE_DOI = "10.5281/zenodo"

# WorldClim 2.1 — 30 arc-second
WORLDCLIM_BIO_URL   = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_bio.zip"
WORLDCLIM_ELEV_URL  = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_elev.zip"
WORLDCLIM_CITATION  = "Fick SE, Hijmans RJ (2017). WorldClim 2. Int. J. Climatology 37:4302-4315."
WORLDCLIM_BANDS     = {1, 4, 7, 12, 15}

# Hengl soil great groups — Zenodo
SOIL_ZENODO_RECORD  = "3528062"
SOIL_ZENODO_DOI     = "10.5281/zenodo.3528062"
SOIL_CITATION       = "Hengl T, Nauman T (2018). Predicted USDA soil great groups at 250m. Zenodo."

# Natural Earth 1:10m — direct S3 URLs (stable, versioned)
BASEMAP_DATASETS = {
    "ne_10m_admin_0_countries":        "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip",
    "ne_10m_admin_1_states_provinces": "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip",
    "ne_10m_land":                     "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
}

CHUNK_SIZE = 1024 * 1024  # 1 MB

# Required Darwin Core columns — must match lore/geo.py:REQUIRED_OCC_COLS
REQUIRED_OCC_COLS = [
    "gbifID",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "verbatimScientificName",
    "eventDate",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _stream_to_memory(url: str, label: str) -> io.BytesIO:
    """Stream a URL into memory with a tqdm progress bar."""
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=label) as pbar:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            buf.write(chunk)
            pbar.update(len(chunk))
    buf.seek(0)
    return buf


def _stream_to_file(url: str, dest: Path, label: str) -> None:
    """Stream a URL directly to a file with a tqdm progress bar."""
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=label) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))


def _zenodo_files(record: str) -> list[dict]:
    """Return the file listing for a Zenodo record."""
    url = f"https://zenodo.org/api/records/{record}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()["files"]


# ---------------------------------------------------------------------------
# Run-specific: occurrences
# ---------------------------------------------------------------------------

def download_occurrences(gbif_doi: str, output_dir: Path) -> None:
    """
    Fetch GBIF occurrence records by DOI or download key and write to
    <output_dir>/occurrences.csv.

    Accepts any of:
        "10.15468/dl.3cv9hy"
        "https://doi.org/10.15468/dl.3cv9hy"
        "https://www.gbif.org/occurrence/download/0059680-260226173443078"
        "0059680-260226173443078"  (raw key)
    """
    dest = output_dir / "occurrences.csv"
    if dest.exists():
        print(f"[occurrences] Already present: {dest}  (delete to re-download)")
        return

    # Normalise to raw key or DOI
    doi = gbif_doi.strip()
    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("https://www.gbif.org/occurrence/download/", "")

    meta_url = f"https://api.gbif.org/v1/occurrence/download/{doi}"
    meta = requests.get(meta_url).json()
    if meta.get("status") != "SUCCEEDED":
        raise RuntimeError(
            f"GBIF download is not ready. Status: {meta.get('status')}\n"
            f"Check: https://www.gbif.org/occurrence/download/{doi}"
        )

    download_url  = meta["downloadLink"]
    n_records     = meta.get("totalRecords", "unknown")
    gbif_citation = (
        f"GBIF.org GBIF Occurrence Download "
        f"https://doi.org/{meta.get('doi', doi)}"
    )
    print(f"[occurrences] {n_records:,} records  —  {gbif_citation}")

    buf = _stream_to_memory(download_url, "Downloading occurrences")

    print("[occurrences] Extracting...")
    with zipfile.ZipFile(buf) as z:
        csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(csv_name) as src:
            dest.write_bytes(src.read())

    print(f"[occurrences] Saved to {dest}\n")


def copy_occurrences(src: Path, output_dir: Path) -> None:
    """
    Copy a user-provided Darwin Core occurrence CSV to
    <output_dir>/occurrences.csv.

    Performs a lightweight preflight check: file exists, is non-empty,
    has a header row, and contains all required Darwin Core columns.

    Full data validation (coordinate ranges, type coercion, taxon matching)
    occurs in lore/geo.py:load_occurrences() at runtime.

    Parameters
    ----------
    src        : Path to the user-provided occurrence CSV or TSV.
    output_dir : Destination directory. File is written as occurrences.csv.
    """
    dest = output_dir / "occurrences.csv"
    if dest.exists():
        print(f"[occurrences] Already present: {dest}  (delete to re-copy)")
        return

    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Occurrences file not found: {src}")
    if src.stat().st_size == 0:
        raise ValueError(f"Occurrences file is empty: {src}")

    # Peek at header and detect separator
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
    if not header_line.strip():
        raise ValueError(f"Occurrences file has no header row: {src}")

    sep = "\t" if "\t" in header_line else ","
    header_cols = [c.strip() for c in header_line.split(sep)]

    missing = [c for c in REQUIRED_OCC_COLS if c not in header_cols]
    if missing:
        # Check for case mismatches to give actionable suggestions
        col_lower = {c.lower(): c for c in header_cols}
        suggestions = []
        for m in missing:
            if m.lower() in col_lower:
                suggestions.append(f"  '{col_lower[m.lower()]}' → rename to '{m}'")
        msg = (
            f"[occurrences] Custom file is missing required Darwin Core columns: "
            f"{missing}\n"
            f"  Required: {REQUIRED_OCC_COLS}"
        )
        if suggestions:
            msg += "\n  Possible fixes (column rename required):\n" + "\n".join(suggestions)
        raise ValueError(msg)

    shutil.copy2(src, dest)
    print(f"[occurrences] Copied {src.name} → {dest}")
    print(f"[occurrences] Full validation occurs at runtime in load_occurrences().\n")


# ---------------------------------------------------------------------------
# Run-specific: MDD range maps
# ---------------------------------------------------------------------------

def list_mdd_groups(zenodo_record: str = MDD_ZENODO_RECORD) -> list[str]:
    """
    Query a MDD Zenodo record and return the available taxonomic group names
    (i.e. the stem of each MDD_<Group>.zip entry).
    """
    files = _zenodo_files(zenodo_record)
    groups = []
    for f in files:
        key = f["key"]
        if key.startswith("MDD_") and key.endswith(".zip"):
            groups.append(key[4:-4])   # strip "MDD_" prefix and ".zip" suffix
    return sorted(groups)


def download_ranges(
    mdd_group: str,
    ranges_dir: Path,
    zenodo_record: str = MDD_ZENODO_RECORD,
) -> None:
    """
    Fetch a MDD taxonomic group range map zip from Zenodo, extract the
    geopackage, and write to <ranges_dir>/MDD_<mdd_group>.gpkg.

    Assumes MDD Zenodo zip/gpkg naming convention:
        zip:  MDD_<Group>.zip
        gpkg: MDD_<Group>.gpkg

    Parameters
    ----------
    mdd_group      : Taxonomic group name as it appears in the Zenodo record,
                     e.g. "Rodentia", "Carnivora", "Chiroptera", "Mammalia".
                     Run --list-mdd-groups to see available options.
    ranges_dir     : Destination directory for the extracted gpkg. Shared
                     across runs -- download once per MDD group.
    zenodo_record  : Zenodo record ID. Default: 6644198 (MDD v1.2).
                     Update for future MDD versions.
    """
    zip_key  = f"MDD_{mdd_group}.zip"
    gpkg_key = f"MDD_{mdd_group}.gpkg"
    dest     = ranges_dir / gpkg_key

    if dest.exists():
        print(f"[ranges] Already present: {dest}  (delete to re-download)")
        return

    ranges_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ranges] Fetching file list from Zenodo record {zenodo_record}...")
    files = _zenodo_files(zenodo_record)

    entry = next((f for f in files if f["key"] == zip_key), None)
    if entry is None:
        available = [f["key"] for f in files if f["key"].startswith("MDD_") and f["key"].endswith(".zip")]
        raise ValueError(
            f"'{zip_key}' not found in Zenodo record {zenodo_record}.\n"
            f"Available groups: {sorted(available)}\n"
            f"Tip: run --list-mdd-groups to see options."
        )

    size_mb      = round(entry["size"] / 1e6, 1)
    zenodo_doi   = f"{MDD_ZENODO_BASE_DOI}.{zenodo_record}"
    mdd_citation = "Marsh et al. (2022) Journal of Mammalogy 103:1-14"
    print(f"[ranges] MDD {mdd_group}  —  {size_mb} MB  —  DOI: {zenodo_doi}")
    print(f"[ranges] {mdd_citation}")

    buf = _stream_to_memory(entry["links"]["self"], f"Downloading MDD {mdd_group}")

    print("[ranges] Extracting (Deflate64)...")
    with zipfile.ZipFile(buf) as z:
        if gpkg_key in z.namelist():
            internal = gpkg_key
        else:
            internal = next((n for n in z.namelist() if n.endswith(".gpkg")), None)
            if internal is None:
                raise RuntimeError(
                    f"No .gpkg found inside {zip_key}. "
                    f"Contents: {z.namelist()}"
                )
        with z.open(internal) as src:
            dest.write_bytes(src.read())

    print(f"[ranges] Saved to {dest}\n")


# ---------------------------------------------------------------------------
# Global: WorldClim rasters
# ---------------------------------------------------------------------------

def download_worldclim(raster_dir: Path) -> None:
    """
    Download WorldClim 2.1 30s elevation and selected bioclimatic bands.
    Skips files that already exist.
    """
    wc_dir = raster_dir / "worldclim"
    wc_dir.mkdir(parents=True, exist_ok=True)

    print(f"[worldclim] {WORLDCLIM_CITATION}")

    # Elevation
    elev_dest = wc_dir / "wc2.1_30s_elev.tif"
    if elev_dest.exists():
        print("[worldclim] Elevation already present — skipping.")
    else:
        print("[worldclim] Downloading elevation (~700 MB zip)...")
        buf = _stream_to_memory(WORLDCLIM_ELEV_URL, "WorldClim 30s elev")
        with zipfile.ZipFile(buf) as z:
            elev_name = next(n for n in z.namelist() if n.endswith(".tif"))
            print(f"[worldclim]   Extracting {elev_name}...")
            elev_dest.write_bytes(z.read(elev_name))
        print(f"[worldclim]   Saved to {elev_dest}")

    # Bioclimatic bands
    missing = [b for b in WORLDCLIM_BANDS
               if not (wc_dir / f"wc2.1_30s_bio_{b}.tif").exists()]
    if not missing:
        print("[worldclim] All bioclim bands already present — skipping.")
    else:
        print(f"[worldclim] Downloading bioclim bands {sorted(WORLDCLIM_BANDS)} (~10.4 GB zip)...")
        buf = _stream_to_memory(WORLDCLIM_BIO_URL, "WorldClim 30s bio")
        print("[worldclim] Extracting selected bands...")
        with zipfile.ZipFile(buf) as z:
            for name in z.namelist():
                if not name.endswith(".tif"):
                    continue
                try:
                    band_num = int(name.split("_")[-1].replace(".tif", ""))
                except ValueError:
                    continue
                if band_num not in WORLDCLIM_BANDS:
                    continue
                dest = wc_dir / f"wc2.1_30s_bio_{band_num}.tif"
                if dest.exists():
                    print(f"[worldclim]   bio_{band_num} already exists — skipping.")
                    continue
                print(f"[worldclim]   Extracting bio_{band_num}...")
                dest.write_bytes(z.read(name))

    print(f"[worldclim] Done. Files in {wc_dir}\n")


# ---------------------------------------------------------------------------
# Global: soil rasters
# ---------------------------------------------------------------------------

def download_soil(raster_dir: Path) -> None:
    """
    Download all 123 USDA soil great group probability rasters from Zenodo.
    Skips files that already exist.
    """
    soil_dir = raster_dir / "soil"
    soil_dir.mkdir(parents=True, exist_ok=True)

    print(f"[soil] Fetching file list from Zenodo record {SOIL_ZENODO_RECORD}...")
    files = _zenodo_files(SOIL_ZENODO_RECORD)

    targets = [
        f for f in files
        if f["key"].endswith(".tif")
        and ("_p_" in f["key"] or "_c_" in f["key"])
        and "_c." not in f["key"]
    ]

    total_size_gb = round(sum(f["size"] for f in targets) / 1e9, 1)
    print(f"[soil] {len(targets)} files  —  ~{total_size_gb} GB total")
    print(f"[soil] {SOIL_CITATION}")
    print(f"[soil] DOI: {SOIL_ZENODO_DOI}\n")

    for entry in targets:
        dest = soil_dir / entry["key"]
        if dest.exists():
            print(f"[soil]   {entry['key']} — already exists, skipping.")
            continue
        size_mb = round(entry["size"] / 1e6, 1)
        _stream_to_file(
            entry["links"]["self"],
            dest,
            f"soil {entry['key'][:40]}... ({size_mb} MB)",
        )

    print(f"[soil] Done. Files in {soil_dir}\n")


# ---------------------------------------------------------------------------
# Global: Natural Earth basemap
# ---------------------------------------------------------------------------

def download_basemap(basemap_dir: Path) -> None:
    """
    Fetch Natural Earth 1:10m cultural and physical vectors and write as
    shapefiles to <basemap_dir>. Skips files that already exist.

    Datasets:
        ne_10m_land/
        ne_10m_admin_0_countries/
        ne_10m_admin_1_states_provinces/

    Source: https://www.naturalearthdata.com
    """
    basemap_dir.mkdir(parents=True, exist_ok=True)
    print("[basemap] Natural Earth 1:10m  —  naturalearthdata.com")

    all_present = all(
        (basemap_dir / name).exists()
        for name in BASEMAP_DATASETS
    )
    if all_present:
        print("[basemap] All files already present — skipping.")
        return

    for name, url in BASEMAP_DATASETS.items():
        dest_dir = basemap_dir / name
        if dest_dir.exists():
            print(f"[basemap]   {name} already present — skipping.")
            continue
        print(f"[basemap]   Downloading {name}...")
        buf = _stream_to_memory(url, f"basemap {name}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(buf) as z:
            z.extractall(dest_dir)
        shps = list(dest_dir.glob("*.shp"))
        if not shps:
            print(f"[basemap]   WARNING: no .shp found in {name} zip.")
        else:
            print(f"[basemap]   Saved {name}/  ({shps[0].name})")

    print(f"[basemap] Done. Files in {basemap_dir}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download all data required for a LORE disambiguation run.\n\n"
            "Occurrence source: provide exactly one of --gbif-doi or\n"
            "--occurrences-file, unless --skip-occurrences is set.\n\n"
            "Run --list-mdd-groups first to see available MDD taxonomic\n"
            "groups for a given Zenodo record."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Discovery ----------------------------------------------------------
    parser.add_argument(
        "--list-mdd-groups",
        action="store_true",
        help=(
            "List available taxonomic groups in the target MDD Zenodo record "
            "and exit. Use with --mdd-zenodo-record to inspect other versions."
        ),
    )

    # ---- Occurrence source (mutually exclusive) ------------------------------
    occ_group = parser.add_mutually_exclusive_group()
    occ_group.add_argument(
        "--gbif-doi",
        default=None,
        metavar="DOI",
        help=(
            "GBIF occurrence download DOI or key. Accepts: "
            "'10.15468/dl.xxxxx', "
            "'https://doi.org/10.15468/dl.xxxxx', "
            "'https://www.gbif.org/occurrence/download/<key>', "
            "or raw numeric key. "
            "Mutually exclusive with --occurrences-file."
        ),
    )
    occ_group.add_argument(
        "--occurrences-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a custom Darwin Core occurrence CSV or TSV. "
            f"Must contain columns: {', '.join(REQUIRED_OCC_COLS)}. "
            "Coordinates must be decimal degrees (EPSG:4326). "
            "Full validation occurs at runtime in load_occurrences(). "
            "Mutually exclusive with --gbif-doi."
        ),
    )

    # ---- MDD group ----------------------------------------------------------
    parser.add_argument(
        "--mdd-group",
        default="Rodentia",
        help=(
            "MDD taxonomic group to download, e.g. 'Rodentia', 'Carnivora', "
            "'Chiroptera', 'Mammalia'. Must match a zip key in the Zenodo record "
            "(MDD_<group>.zip). Run --list-mdd-groups to see all options. "
            "Default: Rodentia."
        ),
    )
    parser.add_argument(
        "--mdd-zenodo-record",
        default=MDD_ZENODO_RECORD,
        help=(
            f"Zenodo record ID for MDD range maps. Default: {MDD_ZENODO_RECORD} "
            "(concept DOI, resolves to latest MDD version). "
            "Pass a specific versioned record ID to pin to a given release."
        ),
    )

    # ---- Output directories -------------------------------------------------
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for run-specific outputs (occurrences.csv). "
            "Required when downloading occurrences (i.e. when --skip-occurrences "
            "is not set). Not needed if only downloading global data. "
            "Example: runs/peromyscus_split_2026"
        ),
    )
    parser.add_argument(
        "--ranges-dir",
        type=Path,
        default=DEFAULT_RANGES_DIR,
        help=f"Directory for MDD range map geopackages. Shared across runs. Default: {DEFAULT_RANGES_DIR}",
    )
    parser.add_argument(
        "--raster-dir",
        type=Path,
        default=DEFAULT_RASTER_DIR,
        help=f"Directory for global raster data. Default: {DEFAULT_RASTER_DIR}",
    )
    parser.add_argument(
        "--basemap-dir",
        type=Path,
        default=DEFAULT_BASEMAP_DIR,
        help=f"Directory for Natural Earth basemap vectors. Default: {DEFAULT_BASEMAP_DIR}",
    )

    # ---- Skip flags ---------------------------------------------------------
    parser.add_argument("--skip-occurrences", action="store_true",
                        help="Skip occurrence acquisition (GBIF or custom file).")
    parser.add_argument("--skip-ranges",      action="store_true",
                        help="Skip MDD range map download.")
    parser.add_argument("--skip-worldclim",   action="store_true",
                        help="Skip WorldClim raster download.")
    parser.add_argument("--skip-soil",        action="store_true",
                        help="Skip soil raster download.")
    parser.add_argument("--skip-basemap",     action="store_true",
                        help="Skip Natural Earth basemap download.")

    args = parser.parse_args()

    # ---- Discovery mode -----------------------------------------------------
    if args.list_mdd_groups:
        print(f"Available MDD groups in Zenodo record {args.mdd_zenodo_record}:")
        for group in list_mdd_groups(args.mdd_zenodo_record):
            print(f"  {group}")
        sys.exit(0)

    # ---- Occurrence source validation ---------------------------------------
    if not args.skip_occurrences:
        if args.gbif_doi is None and args.occurrences_file is None:
            parser.error(
                "One of --gbif-doi or --occurrences-file is required "
                "unless --skip-occurrences is set."
            )
        if args.output_dir is None:
            parser.error(
                "--output-dir is required when downloading occurrences.\n"
                "  Example: --output-dir runs/peromyscus_split_2026\n"
                "  To skip occurrence download: --skip-occurrences"
            )

    # ---- Header -------------------------------------------------------------
    occ_source = (
        f"GBIF DOI: {args.gbif_doi}"          if args.gbif_doi else
        f"File: {args.occurrences_file}"      if args.occurrences_file else
        "skipped"
    )
    print("=" * 55)
    print("  LORE - data acquisition")
    print("=" * 55)
    print(f"  Occurrences : {occ_source}")
    print(f"  MDD group   : {args.mdd_group}")
    print(f"  Zenodo rec  : {args.mdd_zenodo_record}")
    print(f"  Output dir  : {args.output_dir or 'n/a (occurrences skipped)'}")
    print(f"  Ranges dir  : {args.ranges_dir}")
    print(f"  Raster dir  : {args.raster_dir}")
    print(f"  Basemap dir : {args.basemap_dir}")
    print("=" * 55)
    print()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Acquire ------------------------------------------------------------
    if not args.skip_occurrences:
        if args.occurrences_file:
            copy_occurrences(args.occurrences_file, args.output_dir)
        else:
            download_occurrences(args.gbif_doi, args.output_dir)
    else:
        print("[occurrences] Skipped.\n")

    if not args.skip_ranges:
        download_ranges(args.mdd_group, args.ranges_dir, args.mdd_zenodo_record)
    else:
        print("[ranges] Skipped.\n")

    if not args.skip_worldclim:
        download_worldclim(args.raster_dir)
    else:
        print("[worldclim] Skipped.\n")

    if not args.skip_soil:
        download_soil(args.raster_dir)
    else:
        print("[soil] Skipped.\n")

    if not args.skip_basemap:
        download_basemap(args.basemap_dir)
    else:
        print("[basemap] Skipped.\n")

    print("=" * 55)
    print("  Done. Run scripts/preprocess_rasters.py next.")
    print("=" * 55)


if __name__ == "__main__":
    main()
