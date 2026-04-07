"""
lore/visualize.py

Spatial figure generation for LORE (Latent Occurrence Resolution Engine)
disambiguation output.

Produces a single high-resolution PNG map figure showing all occurrence
records coloured by resolved taxon, with visual distinction between
disambiguation methods:

    geo-resolved     : small filled circle, 40% alpha
    ml-resolved      : larger filled circle, full alpha, white edge
    ml_low_confidence: larger filled circle, full alpha, yellow edge
    excluded         : small grey x marker

Range polygons for each destination taxon are drawn as filled polygons
at 30% opacity using the same per-taxon colour.

Basemap layers (Natural Earth 1:10m, clipped to occurrence bbox):
    land, country borders, state/province borders

Colour palette:
    ≤7 taxa : Wong (2011) 8-colour colorblind-safe palette (black omitted)
    >7 taxa : matplotlib tab20 (perceptually distinct; colorblind safety
              not guaranteed — documented limitation)

Usage (library)
---------------
    from lore.visualize import visualize

    visualize(
        disambiguated  = "runs/peromyscus_split_2026/data/disambiguated.csv",
        ranges_file    = "lore/data/Rodentia/MDD_Rodentia.gpkg",
        dest_taxa      = ["Peromyscus maniculatus", "Peromyscus sonoriensis"],
        output         = "runs/peromyscus_split_2026/figures/map.png",
        basemap_dir    = "lore/data/basemap",
    )

Usage (CLI)
-----------
    python -m lore.visualize \\
        --disambiguated  runs/peromyscus_split_2026/data/disambiguated.csv \\
        --ranges-file    lore/data/Rodentia/MDD_Rodentia.gpkg \\
        --dest-taxa      "Peromyscus maniculatus" "Peromyscus sonoriensis" \\
        --output         runs/peromyscus_split_2026/figures/map.png \\
        --basemap-dir    lore/data/basemap \\
        [--scale 1.0] \\
        [--dpi 300] \\
        [--buffer-deg 1.0]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pyproj
from shapely.geometry import box

from lore.geo import load_ranges

matplotlib.use("Agg")  # non-interactive backend — safe for subprocess/HPC calls

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wong (2011) colorblind-safe palette — black (#000000) omitted for legibility
# on dark basemap elements. Order chosen for maximal perceptual separation.
WONG_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

WONG_THRESHOLD = 7   # use Wong below/at this count; tab20 above

# Marker styling
GEO_MARKER_SIZE    = 8
ML_MARKER_SIZE     = 18
GEO_ALPHA          = 0.40
ML_ALPHA           = 1.00
ML_EDGE_COLOR      = "white"
ML_LOW_EDGE_COLOR  = "#F0E442"   # yellow — distinguishable from white
ML_EDGE_WIDTH      = 0.6
EXCL_MARKER        = "x"
EXCL_COLOR         = "#888888"
EXCL_SIZE          = 6
EXCL_ALPHA         = 0.5

RANGE_ALPHA        = 0.30
RANGE_EDGE_ALPHA   = 0.60
RANGE_EDGE_WIDTH   = 0.5

BASEMAP_LAND_COLOR     = "#F5F5F0"
BASEMAP_COUNTRY_COLOR  = "#AAAAAA"
BASEMAP_STATE_COLOR    = "#CCCCCC"
BASEMAP_COUNTRY_WIDTH  = 0.5
BASEMAP_STATE_WIDTH    = 0.3
BASEMAP_OCEAN_COLOR    = "#D6EAF8"

DEFAULT_BUFFER_DEG = 1.0
DEFAULT_SCALE      = 1.0
DEFAULT_DPI        = 300
BASE_FIG_WIDTH     = 10.0   # inches at scale=1.0


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def _build_palette(taxa: list[str]) -> dict[str, str]:
    """
    Assign a colour to each taxon.

    ≤7 taxa → Wong (2011) colorblind-safe palette.
    >7 taxa → matplotlib tab20 (perceptually distinct; not colorblind-safe).
    """
    n = len(taxa)
    if n <= WONG_THRESHOLD:
        colors = WONG_PALETTE[:n]
    else:
        cmap   = matplotlib.colormaps["tab20"]
        colors = [matplotlib.colors.to_hex(cmap(i / n)) for i in range(n)]
        logger.warning(
            "%d taxa exceeds the Wong palette limit (%d). "
            "Falling back to tab20 — colorblind safety not guaranteed.",
            n, WONG_THRESHOLD,
        )
    return dict(zip(taxa, colors))


# ---------------------------------------------------------------------------
# Basemap loading
# ---------------------------------------------------------------------------

def _load_layer(basemap_dir: Path, subdir: str, target_crs) -> gpd.GeoDataFrame:
    """Load a Natural Earth layer from its subdirectory, reproject to target_crs."""
    layer_dir = basemap_dir / subdir
    shps = list(layer_dir.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(
            f"No shapefile found in {layer_dir}. "
            f"Run: python scripts/download_data.py --skip-occurrences "
            f"--skip-ranges --skip-worldclim --skip-soil"
        )
    gdf = gpd.read_file(shps[0])
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(target_crs)


def _clip_to_bbox(gdf: gpd.GeoDataFrame, bbox_proj: tuple) -> gpd.GeoDataFrame:
    """Clip a GeoDataFrame to a projected bounding box tuple (minx,miny,maxx,maxy)."""
    clip_geom = box(*bbox_proj)
    return gdf.clip(clip_geom)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _build_projection(
    occ: pd.DataFrame,
    buffer_deg: float,
    clip_quantile: float = 0.002,   # trims top/bottom 0.2% — removes extreme outliers
    ) -> tuple[str, tuple]:
    lats = occ["decimalLatitude"]
    lons = occ["decimalLongitude"]

    clat = float(lats.median())
    clon = float(lons.median())

    lat_min = float(lats.quantile(clip_quantile))
    lat_max = float(lats.quantile(1 - clip_quantile))
    lon_min = float(lons.quantile(clip_quantile))
    lon_max = float(lons.quantile(1 - clip_quantile))

    proj_crs = (
        f"+proj=aeqd +lat_0={clat:.4f} +lon_0={clon:.4f} "
        f"+datum=WGS84 +units=m +no_defs"
    )

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", proj_crs, always_xy=True
    )

    pad = buffer_deg
    corners_lon = [lon_min - pad, lon_max + pad, lon_min - pad, lon_max + pad]
    corners_lat = [lat_min - pad, lat_min - pad, lat_max + pad, lat_max + pad]
    xs, ys = transformer.transform(corners_lon, corners_lat)

    bbox_proj = (min(xs), min(ys), max(xs), max(ys))
    return proj_crs, bbox_proj


# ---------------------------------------------------------------------------
# Figure sizing
# ---------------------------------------------------------------------------

def _figure_size(bbox_proj: tuple, scale: float) -> tuple[float, float]:
    """
    Compute figure dimensions in inches from the projected bounding box
    aspect ratio, scaled by `scale`.
    """
    minx, miny, maxx, maxy = bbox_proj
    aspect = (maxy - miny) / (maxx - minx)
    width  = BASE_FIG_WIDTH * scale
    height = width * aspect
    return width, height


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def _build_legend(
    palette: dict[str, str],
    has_geo: bool,
    has_ml: bool,
    has_ml_low: bool,
    has_excluded: bool,
) -> list:
    """Build legend handles: taxon colour patches + method markers."""
    handles = []

    # Taxon colour patches
    for taxon, color in palette.items():
        handles.append(mpatches.Patch(color=color, label=taxon, alpha=0.8))

    # Method markers — only show methods present in the data
    handles.append(Line2D([], [], linestyle="none", label=""))  # spacer

    if has_geo:
        handles.append(Line2D(
            [], [], marker="o", color="grey", markersize=6,
            alpha=GEO_ALPHA, linestyle="none",
            label="Geo-resolved",
        ))
    if has_ml:
        handles.append(Line2D(
            [], [], marker="o", color="grey", markersize=9,
            alpha=ML_ALPHA, linestyle="none",
            markeredgecolor=ML_EDGE_COLOR, markeredgewidth=ML_EDGE_WIDTH,
            label="ML-resolved",
        ))
    if has_ml_low:
        handles.append(Line2D(
            [], [], marker="o", color="grey", markersize=9,
            alpha=ML_ALPHA, linestyle="none",
            markeredgecolor=ML_LOW_EDGE_COLOR, markeredgewidth=ML_EDGE_WIDTH,
            label="ML low-confidence",
        ))
    if has_excluded:
        handles.append(Line2D(
            [], [], marker=EXCL_MARKER, color=EXCL_COLOR, markersize=6,
            alpha=EXCL_ALPHA, linestyle="none",
            label="Excluded",
        ))

    return handles


# ---------------------------------------------------------------------------
# Core visualize function
# ---------------------------------------------------------------------------

def visualize(
    disambiguated: str | Path,
    ranges_file:   str | Path,
    dest_taxa:     list[str],
    output:        str | Path,
    basemap_dir:   str | Path,
    buffer_deg:    float = DEFAULT_BUFFER_DEG,
    scale:         float = DEFAULT_SCALE,
    dpi:           int   = DEFAULT_DPI,
) -> Path:
    """
    Generate a spatial disambiguation figure and write to output PNG.

    Parameters
    ----------
    disambiguated : path to disambiguated.csv (output of lore/predict.py)
    ranges_file   : path to MDD .gpkg range file
    dest_taxa     : list of destination taxon names (must match ranges_file)
    output        : output PNG path
    basemap_dir   : directory containing Natural Earth subdirectories
    buffer_deg    : degrees of padding around occurrence bbox. Default: 1.0
    scale         : figure size multiplier. Default: 1.0 (~10 inches wide).
                    Use 2.0 for a larger figure if spatial detail is insufficient.
    dpi           : output resolution in dots per inch. Default: 300

    Returns
    -------
    Path to the written PNG file.
    """
    disambiguated = Path(disambiguated)
    ranges_file   = Path(ranges_file)
    output        = Path(output)
    basemap_dir   = Path(basemap_dir)

    output.parent.mkdir(parents=True, exist_ok=True)

    # ---- load occurrences ---------------------------------------------------
    logger.info("Loading disambiguated occurrences: %s", disambiguated)
    occ = pd.read_csv(disambiguated, low_memory=False)

    required = {"decimalLatitude", "decimalLongitude",
                "final_taxon", "disambiguation_method"}
    missing = required - set(occ.columns)
    if missing:
        raise ValueError(
            f"disambiguated.csv is missing required columns: {missing}. "
            f"Was this file produced by lore/predict.py?"
        )

    occ = occ.dropna(subset=["decimalLatitude", "decimalLongitude"])
    n_total = len(occ)
    logger.info("  %d records loaded", n_total)

    # ---- projection ---------------------------------------------------------
    logger.info("Computing projection from occurrence bounds...")
    proj_crs, bbox_proj = _build_projection(occ, buffer_deg)
    logger.info("  CRS: %s", proj_crs)
    logger.info("  Bbox (proj): %s", [round(x) for x in bbox_proj])

    # ---- reproject occurrences to proj_crs ----------------------------------
    occ_gdf = gpd.GeoDataFrame(
        occ,
        geometry=gpd.points_from_xy(occ["decimalLongitude"], occ["decimalLatitude"]),
        crs="EPSG:4326",
    ).to_crs(proj_crs)

    # ---- load and reproject ranges ------------------------------------------
    logger.info("Loading ranges: %s", ranges_file)
    ranges = load_ranges(ranges_file, dest_taxa)
    ranges = ranges.to_crs(proj_crs)
    ranges = _clip_to_bbox(ranges, bbox_proj)

    # ---- colour palette -----------------------------------------------------
    palette = _build_palette(dest_taxa)

    # ---- load basemap layers ------------------------------------------------
    logger.info("Loading basemap layers...")
    land      = _load_layer(basemap_dir, "ne_10m_land",                    proj_crs)
    countries = _load_layer(basemap_dir, "ne_10m_admin_0_countries",       proj_crs)
    states    = _load_layer(basemap_dir, "ne_10m_admin_1_states_provinces", proj_crs)

    land      = _clip_to_bbox(land,      bbox_proj)
    countries = _clip_to_bbox(countries, bbox_proj)
    states    = _clip_to_bbox(states,    bbox_proj)

    # ---- method masks -------------------------------------------------------
    method      = occ_gdf["disambiguation_method"]
    geo_mask    = method == "geo"
    ml_mask     = method == "ml"
    ml_low_mask = method == "ml_low_confidence"
    excl_mask   = method == "excluded"

    has_geo     = geo_mask.any()
    has_ml      = ml_mask.any()
    has_ml_low  = ml_low_mask.any()
    has_excl    = excl_mask.any()

    # ---- figure setup -------------------------------------------------------
    fig_w, fig_h = _figure_size(bbox_proj, scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_facecolor(BASEMAP_OCEAN_COLOR)
    ax.set_xlim(bbox_proj[0], bbox_proj[2])
    ax.set_ylim(bbox_proj[1], bbox_proj[3])
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- basemap ------------------------------------------------------------
    if not land.empty:
        land.plot(ax=ax, color=BASEMAP_LAND_COLOR, linewidth=0, zorder=1)
    if not states.empty:
        states.plot(ax=ax, color="none",
                    edgecolor=BASEMAP_STATE_COLOR,
                    linewidth=BASEMAP_STATE_WIDTH, zorder=2)
    if not countries.empty:
        countries.plot(ax=ax, color="none",
                       edgecolor=BASEMAP_COUNTRY_COLOR,
                       linewidth=BASEMAP_COUNTRY_WIDTH, zorder=3)

    # ---- range polygons -----------------------------------------------------
    for taxon in dest_taxa:
        color    = palette[taxon]
        taxon_ranges = ranges[ranges["sciname"].str.lower() == taxon.lower()]
        if taxon_ranges.empty:
            logger.warning("No ranges found for taxon: %s", taxon)
            continue
        taxon_ranges.plot(
            ax=ax,
            color=color,
            alpha=RANGE_ALPHA,
            edgecolor=color,
            linewidth=RANGE_EDGE_WIDTH,
            zorder=4,
        )

    # ---- occurrence points — geo-resolved (drawn first, underneath ML) ------
    for taxon in dest_taxa:
        color = palette[taxon]
        mask  = geo_mask & (occ_gdf["final_taxon"] == taxon)
        sub   = occ_gdf[mask]
        if sub.empty:
            continue
        ax.scatter(
            sub.geometry.x, sub.geometry.y,
            c=color, s=GEO_MARKER_SIZE,
            alpha=GEO_ALPHA,
            linewidths=0,
            zorder=5,
            rasterized=True,   # rasterize dense point layers for file size
        )

    # ---- occurrence points — ML-resolved ------------------------------------
    for taxon in dest_taxa:
        color = palette[taxon]
        for mask, edge_color in [
            (ml_mask,     ML_EDGE_COLOR),
            (ml_low_mask, ML_LOW_EDGE_COLOR),
        ]:
            sub = occ_gdf[mask & (occ_gdf["final_taxon"] == taxon)]
            if sub.empty:
                continue
            ax.scatter(
                sub.geometry.x, sub.geometry.y,
                c=color, s=ML_MARKER_SIZE,
                alpha=ML_ALPHA,
                edgecolors=edge_color,
                linewidths=ML_EDGE_WIDTH,
                zorder=6,
                rasterized=True,
            )

    # ---- excluded points ----------------------------------------------------
    if has_excl:
        sub = occ_gdf[excl_mask]
        ax.scatter(
            sub.geometry.x, sub.geometry.y,
            c=EXCL_COLOR, s=EXCL_SIZE,
            alpha=EXCL_ALPHA,
            marker=EXCL_MARKER,
            linewidths=0.4,
            zorder=7,
            rasterized=True,
        )

    # ---- legend -------------------------------------------------------------
    handles = _build_legend(palette, has_geo, has_ml, has_ml_low, has_excl)
    legend  = ax.legend(
        handles=handles,
        loc="best",
        fontsize=7,
        framealpha=0.85,
        edgecolor="#CCCCCC",
        borderpad=0.6,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    legend.get_frame().set_linewidth(0.5)

    # ---- summary annotation -------------------------------------------------
    n_geo    = int(geo_mask.sum())
    n_ml     = int(ml_mask.sum())
    n_ml_low = int(ml_low_mask.sum())
    n_excl   = int(excl_mask.sum())

    summary = (
        f"n={n_total:,}  |  "
        f"geo={n_geo:,}  ml={n_ml:,}"
        + (f"  ml_low={n_ml_low:,}" if n_ml_low else "")
        + (f"  excl={n_excl:,}"     if n_excl   else "")
    )
    ax.annotate(
        summary,
        xy=(0.99, 0.01), xycoords="axes fraction",
        ha="right", va="bottom",
        fontsize=6, color="#444444",
    )

    # ---- save ---------------------------------------------------------------
    logger.info("Writing figure: %s  (%.1f x %.1f in, %d dpi)", output, fig_w, fig_h, dpi)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=BASEMAP_OCEAN_COLOR)
    plt.close(fig)

    print(f"[visualize] Figure written: {output}")
    print(f"[visualize] Size: {fig_w:.1f} x {fig_h:.1f} in at {dpi} DPI")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a spatial disambiguation figure from LORE output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--disambiguated", type=Path, required=True,
                   help="Path to disambiguated.csv (output of lore/predict.py).")
    p.add_argument("--ranges-file", type=Path, required=True,
                   help="Path to MDD range map geopackage (.gpkg).")
    p.add_argument("--dest-taxa", nargs="+", required=True,
                   help="Destination taxon names to plot ranges for.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output PNG path.")
    p.add_argument("--basemap-dir", type=Path,
                   default=Path("lore/data/basemap"),
                   help="Directory containing Natural Earth subdirectories.")
    p.add_argument("--buffer-deg", type=float, default=DEFAULT_BUFFER_DEG,
                   help="Degrees of padding around occurrence bbox.")
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                   help=(
                       "Figure size multiplier. Default 1.0 ≈ 10 inches wide. "
                       "Increase (e.g. 2.0) if spatial detail is insufficient "
                       "at the default size."
                   ))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                   help="Output resolution in dots per inch.")
    p.add_argument("--clip-quantile", type=float, default=0.0005,
                   help="Occurrence centroid clipping quantile..")
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

    print("=" * 55)
    print("  LORE — figure generation")
    print("=" * 55)
    print(f"  Disambiguated : {args.disambiguated}")
    print(f"  Ranges file   : {args.ranges_file}")
    print(f"  Dest taxa     : {args.dest_taxa}")
    print(f"  Output        : {args.output}")
    print(f"  Basemap dir   : {args.basemap_dir}")
    print(f"  Scale         : {args.scale}")
    print(f"  DPI           : {args.dpi}")
    print("=" * 55)

    visualize(
        disambiguated = args.disambiguated,
        ranges_file   = args.ranges_file,
        dest_taxa     = args.dest_taxa,
        output        = args.output,
        basemap_dir   = args.basemap_dir,
        buffer_deg    = args.buffer_deg,
        scale         = args.scale,
        dpi           = args.dpi,
    )


if __name__ == "__main__":
    main()
