"""
scripts/preprocess_rasters.py

Clips global environmental rasters to the bounding box of a set of
occurrence records and produces a run-tagged cache directory of regional
rasters ready for feature extraction.

For soil data, clips each of the 123 probability rasters individually to
the bbox and writes them as separate tifs in a soil/ subdirectory. This
avoids loading a full stacked array into RAM. features.py samples each
raster independently at runtime.

Derives slope from WorldClim elevation using a numpy gradient approach.

Outputs are written to:
    <output-dir>/<run-tag>/cache/
        bbox.json
        elevation.tif
        slope.tif
        soil/
            <class_name>.tif     (one per soil great group, 123 total)
        soil_class_names.json    ordered list of class names
        soil_stats.json          per-class min/max for normalisation
        wc2.1_30s_bio_1.tif
        wc2.1_30s_bio_4.tif
        wc2.1_30s_bio_7.tif
        wc2.1_30s_bio_12.tif
        wc2.1_30s_bio_15.tif

Processing is skipped for files that already exist unless --force is passed.

Usage:
    python scripts/download_data.py \\
        --run-tag peromyscus_split_2026 \\
        --occurrences path/to/geo_disambiguated.parquet \\
        --raster-dir lore/data/rasters \\
        --output-dir runs \\
        [--buffer-deg 1.0] \\
        [--workers 4] \\
        [--force]
"""

import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE         = Path(__file__).parent
PROJECT_ROOT = HERE.parent

DEFAULT_RASTER_DIR = PROJECT_ROOT / "lore" / "data" / "rasters"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"
DEFAULT_BUFFER_DEG = 1.0
DEFAULT_WORKERS    = 4

WORLDCLIM_BANDS = [1, 4, 7, 12, 15]
BBOX_JSON_NAME  = "bbox.json"
BLOCK_SIZE      = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_occurrence_bbox(
    occ_path: Path,
    buffer_deg: float,
) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) padded by buffer_deg degrees."""
    gdf = gpd.read_parquet(occ_path)
    b = gdf.total_bounds
    return (
        float(max(-180.0, b[0] - buffer_deg)),
        float(max(-90.0,  b[1] - buffer_deg)),
        float(min(180.0,  b[2] + buffer_deg)),
        float(min(90.0,   b[3] + buffer_deg)),
    )


def _validate_bbox(run_dir: Path, bbox: tuple, force: bool) -> None:
    """
    Warn if stored bbox differs from current bbox.
    Skipped when force=True -- the cache is about to be regenerated anyway.
    """
    if force:
        return
    bbox_file = run_dir / BBOX_JSON_NAME
    if not bbox_file.exists():
        return
    stored = tuple(json.loads(bbox_file.read_text())["bbox"])
    if not all(abs(a - b) < 0.01 for a, b in zip(stored, bbox)):
        stale = [
            str(p) for p in run_dir.rglob("*.tif")
        ]
        stale_summary = (
            f"{len(stale)} tif(s) in {run_dir}"
            if stale else "no tifs found in cache"
        )
        warnings.warn(
            f"Stored bbox {stored} differs from current bbox {bbox}. "
            f"Cached rasters may be stale: {stale_summary}. "
            f"Re-run with --force to regenerate."
        )

def _write_bbox(run_dir: Path, bbox: tuple) -> None:
    (run_dir / BBOX_JSON_NAME).write_text(json.dumps(
        {"bbox": list(bbox), "description": "minx, miny, maxx, maxy in EPSG:4326"},
        indent=2,
    ))


def _clip_raster_to_bbox(
    src_path: str | Path,
    dest: Path,
    bbox: tuple[float, float, float, float],
    label: str,
    force: bool = False,
) -> None:
    """Read a bbox window from src and write as a clipped, tiled GeoTiff."""
    if dest.exists() and not force:
        print(f"  [{label}] Already exists — skipping.")
        return

    minx, miny, maxx, maxy = bbox
    with rasterio.open(str(src_path)) as src:
        window    = from_bounds(minx, miny, maxx, maxy, src.transform)
        data      = src.read(window=window)
        transform = src.window_transform(window)
        profile   = src.profile.copy()
        profile.update({
            "driver":     "GTiff",
            "height":     data.shape[1],
            "width":      data.shape[2],
            "transform":  transform,
            "compress":   "lzw",
            "tiled":      True,
            "blockxsize": BLOCK_SIZE,
            "blockysize": BLOCK_SIZE,
        })

    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(data)
    print(f"  [{label}] Written: {dest.name}  shape={data.shape}")


def _clip_rasters_concurrent(
    tasks: list[tuple],
    bbox: tuple,
    force: bool,
    max_workers: int,
) -> None:
    """
    Clip multiple rasters to bbox concurrently.
    tasks: list of (src_path, dest, label) tuples.
    """
    pending = [(src, dest, label) for src, dest, label in tasks
               if not Path(dest).exists() or force]
    if not pending:
        print("  All files already exist — skipping.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_clip_raster_to_bbox, src, dest, bbox, label, force): label
            for src, dest, label in pending
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [{label}] ERROR: {e}")


def _derive_slope(elevation_path: Path, dest: Path, force: bool = False) -> None:
    """
    Derive slope (degrees) from elevation using numpy gradient.
    Assumes geographic CRS; approximates metres per degree at mid-latitude.
    """
    if dest.exists() and not force:
        print(f"  [slope] Already exists — skipping.")
        return

    with rasterio.open(elevation_path) as src:
        elev    = src.read(1).astype(np.float32)
        nodata  = src.nodata
        res_deg = src.res[0]
        profile = src.profile.copy()

    res_m = res_deg * 111_320.0

    if nodata is not None:
        elev[elev == nodata] = np.nan

    dy, dx = np.gradient(elev, res_m, res_m)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope  = np.nan_to_num(slope, nan=0.0).astype(np.float32)

    profile.update({
        "dtype":      "float32",
        "count":      1,
        "nodata":     -9999.0,
        "compress":   "lzw",
        "tiled":      True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
    })
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(slope[np.newaxis, ...])
    print(f"  [slope] Written: {dest.name}  shape={slope.shape}")


def _build_soil_cache(
    soil_dir: Path,
    run_dir: Path,
    bbox: tuple[float, float, float, float],
    force: bool,
) -> None:
    """
    Clip each soil probability raster individually to bbox and write to
    <run_dir>/soil/<class_name>.tif. Each task writes its own file so
    concurrent execution is safe. Peak RAM is one band per worker.

    Also writes:
        soil_class_names.json  ordered list of class names (sorted, deterministic)
        soil_stats.json        per-class {"min": ..., "max": ...}
                               for global normalisation at inference time
    """
    soil_cache = run_dir / "soil"
    names_dest = run_dir / "soil_class_names.json"
    stats_dest = run_dir / "soil_stats.json"

    soil_cache.mkdir(parents=True, exist_ok=True)

    prob_tifs = sorted([
        f for f in soil_dir.iterdir()
        if f.suffix == ".tif" and "_p_" in f.name
    ])

    if not prob_tifs:
        raise FileNotFoundError(
            f"No soil probability tifs found in {soil_dir}. "
            f"Run scripts/download_rasters.py first."
        )

    existing = list(soil_cache.glob("*.tif"))
    if len(existing) == len(prob_tifs) and names_dest.exists() and not force:
        print(f"  [soil] {len(existing)} clipped tifs already present — skipping.")
        return

    print(f"  [soil] Clipping {len(prob_tifs)} rasters ...")
    minx, miny, maxx, maxy = bbox

    def _clip_one(tif: Path) -> tuple[str, float, float]:
        """Clip one soil tif to bbox, write result. Returns (class_name, min, max)."""
        parts      = tif.stem.split(".")
        class_name = parts[2] if len(parts) > 2 else tif.stem
        dest       = soil_cache / f"{class_name}.tif"

        if dest.exists() and not force:
            # still need min/max — read from existing file
            with rasterio.open(dest) as src:
                band = src.read(1).astype(np.float32)
            return class_name, float(band.min()), float(band.max())

        with rasterio.open(tif) as src:
            window    = from_bounds(minx, miny, maxx, maxy, src.transform)
            band      = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
            profile   = src.profile.copy()
            profile.update({
                "driver":     "GTiff",
                "dtype":      "float32",
                "count":      1,
                "height":     band.shape[0],
                "width":      band.shape[1],
                "transform":  transform,
                "compress":   "lzw",
                "tiled":      True,
                "blockxsize": BLOCK_SIZE,
                "blockysize": BLOCK_SIZE,
                "nodata":     None,
            })

        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(band[np.newaxis, ...])

        return class_name, float(band.min()), float(band.max())

    results = {}
    for tif in tqdm(prob_tifs, desc="  soil clips", unit="raster"):
        try:
            results[tif] = _clip_one(tif)
        except Exception as e:
            print(f"  [soil] ERROR clipping {tif.name}: {e}")

    # reconstruct in sorted order for deterministic class index
    class_names = []
    band_stats  = {}
    for tif in prob_tifs:
        if tif in results:
            class_name, bmin, bmax = results[tif]
            class_names.append(class_name)
            band_stats[class_name] = {"min": bmin, "max": bmax}

    names_dest.write_text(json.dumps(class_names, indent=2))
    stats_dest.write_text(json.dumps(band_stats, indent=2))

    print(f"  [soil] {len(class_names)} tifs written to {soil_cache}")
    print(f"  [soil] Class names: {names_dest.name}")
    print(f"  [soil] Band stats:  {stats_dest.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clip global rasters to occurrence bbox for a named LORE run."
    )
    parser.add_argument(
        "--run-tag", required=True,
        help="Identifier for this run (e.g. 'peromyscus_split_2026').",
    )
    parser.add_argument(
        "--occurrences", type=Path, required=True,
        help="Path to geo-disambiguated occurrence parquet.",
    )
    parser.add_argument(
        "--raster-dir", type=Path, default=DEFAULT_RASTER_DIR,
        help=f"Directory containing downloaded global rasters. Default: {DEFAULT_RASTER_DIR}",
    )
    parser.add_argument(
    "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    help="Base output directory. Run cache is written to <output-dir>/<run-tag>/cache/. "
         f"Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--buffer-deg", type=float, default=DEFAULT_BUFFER_DEG,
        help=f"Degrees of bbox padding. Default: {DEFAULT_BUFFER_DEG}",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of concurrent I/O workers. Default: {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing cached rasters.",
    )
    args = parser.parse_args()

    run_dir = args.output_dir / args.run_tag / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  LORE — raster preprocessing")
    print("=" * 55)
    print(f"  Run tag:      {args.run_tag}")
    print(f"  Occurrences:  {args.occurrences}")
    print(f"  Raster dir:   {args.raster_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Cache dir:    {run_dir}")
    print(f"  Workers:      {args.workers}")
    print(f"  Force:        {args.force}")
    print("=" * 55)
    print()

    # ---- bbox ---------------------------------------------------------------
    print("Computing occurrence bounding box...")
    bbox = _get_occurrence_bbox(args.occurrences, args.buffer_deg)
    print(f"  bbox (minx, miny, maxx, maxy): {[round(x, 4) for x in bbox]}")
    _validate_bbox(run_dir, bbox, force=args.force)

    wc_dir = args.raster_dir / "worldclim"

    # ---- elevation + bioclim clipped concurrently ---------------------------
    clip_tasks = []

    wc_elev = wc_dir / "wc2.1_30s_elev.tif"
    if wc_elev.exists():
        clip_tasks.append((wc_elev, run_dir / "elevation.tif", "elevation"))
    else:
        print(f"\n[elevation] Source not found: {wc_elev} — skipping.")

    for band in WORLDCLIM_BANDS:
        src = wc_dir / f"wc2.1_30s_bio_{band}.tif"
        if src.exists():
            clip_tasks.append((src, run_dir / f"wc2.1_30s_bio_{band}.tif", f"bio_{band}"))
        else:
            print(f"\n[bio_{band}] Source not found: {src} — skipping.")

    if clip_tasks:
        print(f"\nClipping {len(clip_tasks)} WorldClim rasters ({args.workers} workers)...")
        _clip_rasters_concurrent(clip_tasks, bbox, args.force, args.workers)

    # ---- slope --------------------------------------------------------------
    print("\nDeriving slope from elevation...")
    elev_out = run_dir / "elevation.tif"
    if not elev_out.exists():
        print("  [slope] Elevation not available — skipping.")
    else:
        _derive_slope(elev_out, run_dir / "slope.tif", args.force)

    # ---- soil individual tifs -----------------------------------------------
    print("\nClipping soil probability rasters...")
    soil_dir = args.raster_dir / "soil"
    if not soil_dir.exists():
        print(f"  [soil] Soil directory not found: {soil_dir} — skipping.")
    else:
        _build_soil_cache(soil_dir, run_dir, bbox, args.force)

    # ---- write bbox record --------------------------------------------------
    # Only written on a fresh run or forced regeneration. A no-op run (all
    # files already exist) must not overwrite the stored bbox, as that would
    # silently mask stale cache warnings on subsequent runs.
    bbox_file = run_dir / BBOX_JSON_NAME
    if args.force or not bbox_file.exists():
        _write_bbox(run_dir, bbox)
        print(f"\nBbox record written to {run_dir / BBOX_JSON_NAME}")
    else:
        print(f"\nBbox record unchanged: {bbox_file}")

    print()
    print("=" * 55)
    print(f"  Done. Cache ready at: {run_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
