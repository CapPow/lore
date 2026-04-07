"""
run_pipeline.py

End-to-end orchestration script for a LORE (Latent Occurrence Resolution
Engine) taxonomic disambiguation run.

Executes the full pipeline in order:
    1. Data acquisition          -- scripts/download_data.py
    2. Geospatial disambiguation -- lore/geo.py
    3. Raster preprocessing      -- scripts/preprocess_rasters.py
    4. Feature extraction        -- lore/features.py
    5. Feature analysis          -- lore/analysis.py
    6. Model training            -- lore/model.py
    7. Inference                 -- lore/predict.py
    8. Figure generation         -- lore/visualize.py

Each step checks whether its primary output already exists and skips if so,
unless --force is passed. This makes iterative runs (e.g. re-training with
different --exclude-features after reviewing the analysis report) fast.

Iterative workflow
------------------
On a first run the analysis report is written to <data-dir>/analysis/.
Review analysis_report.txt, then re-run with --exclude-features if desired.
Steps 1-5 will be skipped (outputs exist), only training and inference rerun.

Usage
-----
    python run_pipeline.py \\
        --run-tag peromyscus_split_2026 \\
        --gbif-doi 10.15468/dl.3cv9hy \\
        --source-taxa "Peromyscus maniculatus" \\
        --dest-taxa "Peromyscus maniculatus" "Peromyscus sonoriensis" \\
                    "Peromyscus gambelii" "Peromyscus keeni" \\
                    "Peromyscus labecula" "Peromyscus arcticus" \\
        [--data-dir runs/peromyscus_split_2026] \\
        [--ranges-dir lore/data/ranges] \\
        [--raster-dir lore/data/rasters] \\
        [--exclude-features feat_lat feat_lon] \\
        [--confidence-threshold 0.8] \\
        [--workers 8] \\
        [--dropout 0.1] \\
        [--device auto] \\
        [--force] \\
        [--skip-download] \\
        [--skip-rasters]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
HERE         = Path(__file__).parent.resolve()
PROJECT_ROOT = HERE
PYTHON       = sys.executable  # use the same interpreter that launched this script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)


def _step(n: int, title: str) -> None:
    print()
    _sep(f"Step {n}: {title}")


def _skip(msg: str) -> None:
    print(f"  [skip] {msg}")


def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess command, raising on non-zero exit."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{label}' failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}"
        )


def _resolve_device(device_str: str) -> str:
    """Resolve device string, warning if CPU."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            warnings.warn(
                "No GPU detected -- training will run on CPU and may be slow. "
                "Consider running on a machine with CUDA or MPS support.",
                stacklevel=2,
            )
            return "cpu"
    if device_str == "cpu":
        warnings.warn(
            "Device explicitly set to CPU -- training may be slow.",
            stacklevel=2,
        )
    return device_str


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_download(
    gbif_doi: str | None,
    occurrences_file: Path | None,
    mdd_group: str,
    output_dir: Path,
    ranges_dir: Path,
    raster_dir: Path,
    basemap_dir: Path,
    skip: bool,
    skip_rasters: bool,
    occ_out: Path,
    ranges_out: Path,
    force: bool,
) -> None:
    _step(1, "Data acquisition")
    if skip:
        _skip("Download skipped (--skip-download).")
        return

    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "download_data.py"),
        "--mdd-group",   mdd_group,
        "--output-dir",  str(output_dir),
        "--ranges-dir",  str(ranges_dir),
        "--raster-dir",  str(raster_dir),
        "--basemap-dir", str(basemap_dir),
    ]

    if occurrences_file:
        cmd += ["--occurrences-file", str(occurrences_file)]
    elif gbif_doi:
        cmd += ["--gbif-doi", gbif_doi]

    if occ_out.exists() and not force:
        cmd.append("--skip-occurrences")
    if ranges_out.exists() and not force:
        cmd.append("--skip-ranges")
    if skip_rasters:
        cmd += ["--skip-worldclim", "--skip-soil", "--skip-basemap", "--skip-landcover"]

    _run(cmd, "download_data")


def step_geo(
    occ_path: Path,
    ranges_file: Path,
    source_taxa: list[str],
    dest_taxa: list[str],
    run_tag: str,
    output_dir: Path,
    geo_out: Path,
    workers: int,
    force: bool,
) -> None:
    _step(2, "Geospatial disambiguation")

    if geo_out.exists() and not force:
        _skip(f"Geo-disambiguated parquet already present: {geo_out}")
        return

    geo_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, "-m", "lore.geo",
        "--occurrences",  str(occ_path),
        "--ranges-file",  str(ranges_file),
        "--run-tag",      run_tag,
        "--output-dir",   str(output_dir),
        "--output",       str(geo_out),
        "--workers",      str(workers),
    ]
    
    cmd += ["--source-taxa"] + source_taxa
    cmd += ["--dest-taxa"] + dest_taxa

    _run(cmd, "lore.geo")


def step_preprocess(
    run_tag: str,
    geo_out: Path,
    raster_dir: Path,
    output_dir: Path,
    workers: int,
    force: bool,
) -> None:
    _step(3, "Raster preprocessing")

    bbox_file = output_dir / run_tag / "cache" / "bbox.json"
    if bbox_file.exists() and not force:
        _skip(f"Raster cache already present: {output_dir / run_tag / 'cache'}")
        return

    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "preprocess_rasters.py"),
        "--run-tag",     run_tag,
        "--occurrences", str(geo_out),
        "--raster-dir",  str(raster_dir),
        "--output-dir",  str(output_dir),
        "--workers",     str(workers),
    ]
    if force:
        cmd.append("--force")
    _run(cmd, "preprocess_rasters")


def step_features(
    geo_out: Path,
    run_tag: str,
    output_dir: Path,
    features_out: Path,
    workers: int,
    force: bool,
) -> None:
    _step(4, "Feature extraction")

    if features_out.exists() and not force:
        _skip(f"Features parquet already present: {features_out}")
        return

    cmd = [
        PYTHON, "-m", "lore.features",
        "--occurrences", str(geo_out),
        "--run-tag",     run_tag,
        "--output-dir",  str(output_dir),
        "--output",      str(features_out),
        "--workers",     str(workers),
    ]
    _run(cmd, "lore.features")


def step_analysis(
    features_out: Path,
    analysis_dir: Path,
    run_tag: str,
    workers: int,
    force: bool,
) -> None:
    _step(5, "Feature analysis")

    report_path = analysis_dir / "analysis_report.txt"
    if report_path.exists() and not force:
        _skip(f"Analysis report already present: {report_path}")
        print(f"\n  To review: cat {report_path}")
        print(f"  To re-run with exclusions: add --exclude-features and --force")
        return

    cmd = [
        PYTHON, "-m", "lore.analysis",
        "--features",   str(features_out),
        "--output-dir", str(analysis_dir),
        "--run-tag",    run_tag,
        "--workers",    str(workers),
    ]
    _run(cmd, "lore.analysis")

    print()
    print(f"  Analysis report: {report_path}")
    print(f"  Review the report, then re-run with --exclude-features if needed.")
    print(f"  Steps 1-5 will be skipped on re-run (outputs exist).")


def step_train(
    features_out: Path,
    run_tag: str,
    output_dir: Path,
    exclude_features: list[str],
    confidence_threshold: float | None,
    dropout: float,
    device: str,
    force: bool,
) -> Path:
    _step(6, "Model training")

    checkpoint_path = output_dir / run_tag / "cache" / "model" / "checkpoint.pt"
    if checkpoint_path.exists() and not force:
        _skip(f"Checkpoint already present: {checkpoint_path}")
        return checkpoint_path

    cmd = [
        PYTHON, "-m", "lore.model",
        "--features",   str(features_out),
        "--run-tag",    run_tag,
        "--output-dir", str(output_dir),
        "--device",     device,
    ]
    if exclude_features:
        cmd += ["--exclude-features"] + exclude_features
    if confidence_threshold is not None:
        cmd += ["--confidence-threshold", str(confidence_threshold)]
    cmd += ["--dropout", str(dropout)]

    _run(cmd, "lore.model")
    return checkpoint_path


def step_predict(
    features_out: Path,
    checkpoint_path: Path,
    output_csv: Path,
    confidence_threshold: float | None,
    device: str,
    force: bool,
) -> None:
    _step(7, "Inference and final output")

    if output_csv.exists() and not force:
        if checkpoint_path.exists() and checkpoint_path.stat().st_mtime > output_csv.stat().st_mtime:
            pass  # checkpoint is newer -- re-run predict even without --force
        else:
            _skip(f"Output CSV already present: {output_csv}")
            return

    cmd = [
        PYTHON, "-m", "lore.predict",
        "--features",   str(features_out),
        "--checkpoint", str(checkpoint_path),
        "--output",     str(output_csv),
        "--device",     device,
    ]
    if confidence_threshold is not None:
        cmd += ["--confidence-threshold", str(confidence_threshold)]

    _run(cmd, "lore.predict")


def step_visualize(
    output_csv: Path,
    ranges_file: Path,
    dest_taxa: list[str],
    figures_dir: Path,
    basemap_dir: Path,
    force: bool,
) -> Path:
    _step(8, "Figure generation")

    figure_out = figures_dir / "map.png"
    if figure_out.exists() and not force:
        _skip(f"Figure already present: {figure_out}")
        return figure_out

    figures_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, "-m", "lore.visualize",
        "--disambiguated", str(output_csv),
        "--ranges-file",   str(ranges_file),
        "--dest-taxa",     *dest_taxa,
        "--output",        str(figure_out),
        "--basemap-dir",   str(basemap_dir),
    ]
    _run(cmd, "lore.visualize")
    return figure_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end LORE taxonomic disambiguation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- required ----------------------------------------------------------
    parser.add_argument("--run-tag", required=True,
                        help="Identifier for this run. Used as cache subdirectory "
                             "and model output tag. E.g. 'peromyscus_split_2026'.")
    parser.add_argument("--gbif-doi", default=None,
                        help="GBIF occurrence download DOI or key. "
                             "Mutually exclusive with --occurrences-file.")
    parser.add_argument("--occurrences-file", type=Path, default=None,
                        help="Path to a custom Darwin Core occurrence CSV. "
                             "Mutually exclusive with --gbif-doi.")
    parser.add_argument("--source-taxa", nargs="+", required=True,
                        help="Pre-split taxon name(s) to filter occurrences. "
                             "Accepts genus or binomial. "
                             "E.g. --source-taxa 'Peromyscus maniculatus'")
    parser.add_argument("--dest-taxa", nargs="+", required=True,
                        help="Post-split destination taxon names for range loading. "
                             "E.g. --dest-taxa 'Peromyscus sonoriensis' "
                             "'Peromyscus gambelii' ...")
    parser.add_argument("--mdd-group", default="Rodentia",
                        help="MDD taxonomic group to download. "
                             "Run: python scripts/download_data.py --list-mdd-groups. "
                             "Default: Rodentia.")

    # ---- paths -------------------------------------------------------------
    parser.add_argument("--data-dir", type=Path,
                        default=None,
                        help="Directory for run-specific outputs (occurrences, parquet, "
                             "features, model outputs, figures). "
                             "Defaults to runs/<run-tag> if not specified.")
    parser.add_argument("--ranges-dir", type=Path,
                        default=PROJECT_ROOT / "lore" / "data" / "ranges",
                        help="Directory for MDD range map geopackages. "
                             "Shared across runs — downloaded once per MDD group.")
    parser.add_argument("--raster-dir", type=Path,
                        default=PROJECT_ROOT / "lore" / "data" / "rasters",
                        help="Directory containing downloaded global rasters.")
    parser.add_argument("--basemap-dir", type=Path,
                        default=PROJECT_ROOT / "lore" / "data" / "basemap",
                        help="Directory for Natural Earth basemap vectors.")

    # ---- pipeline controls -------------------------------------------------
    parser.add_argument("--exclude-features", nargs="*", default=[],
                        help="feat_* column names to exclude from model training. "
                             "Obtain candidates from analysis_report.txt. "
                             "E.g. --exclude-features feat_lat feat_lon")
    parser.add_argument("--confidence-threshold", type=float, default=None,
                        help="Softmax probability below which ML predictions are "
                             "flagged ml_low_confidence. Default: None (all assigned). "
                             "Confidence scores always written to output.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for geo disambiguation, raster "
                             "sampling, and feature analysis.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for model training. Inversely correlated "
                            "with dataset size -- use 0.1 for large datasets "
                            "(~60k+ records) and 0.2-0.25 for small datasets "
                            "(~5k records). See docs/hyperparameter_sweep.md.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for model training and inference. "
                             "'auto' selects CUDA > MPS > CPU and warns if CPU.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs exist.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip GBIF and MDD range map downloads.")
    parser.add_argument("--skip-rasters", action="store_true",
                        help="Skip global raster downloads (EarthEnv + WorldClim + soil). "
                             "Assumes raster files are already present.")

    args = parser.parse_args()

    # ---- mutual exclusivity check ------------------------------------------
    if args.gbif_doi and args.occurrences_file:
        parser.error("--gbif-doi and --occurrences-file are mutually exclusive.")
    if not args.skip_download and not args.gbif_doi and not args.occurrences_file:
        parser.error(
            "One of --gbif-doi or --occurrences-file is required "
            "unless --skip-download is set."
        )

    # ---- resolve device before header print --------------------------------
    device = _resolve_device(args.device)

    # ---- resolve paths -----------------------------------------------------
    data_dir     = args.data_dir or PROJECT_ROOT / "runs" / args.run_tag
    base_dir     = data_dir.parent           # e.g. runs/
    ranges_out   = args.ranges_dir / f"MDD_{args.mdd_group}.gpkg"
    occ_out      = data_dir / "occurrences.csv"
    geo_out      = data_dir / "geo_disambiguated.parquet"
    features_out = data_dir / "features.parquet"
    analysis_dir = data_dir / "analysis"
    output_csv   = data_dir / "disambiguated.csv"
    figures_dir  = data_dir / "figures"
    cache_dir    = data_dir / "cache"        # for checkpoint path derivation only

    # ---- header ------------------------------------------------------------
    print()
    _sep()
    print("  LORE -- end-to-end disambiguation pipeline")
    _sep()
    print(f"  Run tag          : {args.run_tag}")
    print(f"  Source taxa      : {args.source_taxa}")
    print(f"  Dest taxa        : {args.dest_taxa}")
    print(f"  GBIF DOI         : {args.gbif_doi}")
    print(f"  Occurrences file : {args.occurrences_file or 'from GBIF DOI'}")
    print(f"  Ranges file      : {ranges_out}")
    print(f"  Data dir         : {data_dir}")
    print(f"  Base dir         : {base_dir}")
    print(f"  Cache dir        : {cache_dir}")
    print(f"  Raster dir       : {args.raster_dir}")
    print(f"  Excluded features: {args.exclude_features or 'none'}")
    print(f"  Conf. threshold  : {args.confidence_threshold or 'none'}")
    print(f"  Workers          : {args.workers}")
    print(f"  Device           : {device}")
    print(f"  Force            : {args.force}")
    _sep()

    # ---- run pipeline ------------------------------------------------------
    step_download(
        gbif_doi         = args.gbif_doi,
        occurrences_file = args.occurrences_file,
        mdd_group        = args.mdd_group,
        output_dir       = data_dir,
        ranges_dir       = args.ranges_dir,
        raster_dir       = args.raster_dir,
        basemap_dir      = args.basemap_dir,
        skip             = args.skip_download,
        skip_rasters     = args.skip_rasters,
        occ_out          = occ_out,
        ranges_out       = ranges_out,
        force            = args.force,
    )

    step_geo(
        occ_path    = occ_out,
        ranges_file = ranges_out,
        source_taxa = args.source_taxa,
        dest_taxa   = args.dest_taxa,
        run_tag     = args.run_tag,
        output_dir  = base_dir,
        geo_out     = geo_out,
        workers     = args.workers,
        force       = args.force,
    )

    step_preprocess(
        run_tag    = args.run_tag,
        geo_out    = geo_out,
        raster_dir = args.raster_dir,
        output_dir = base_dir,
        workers    = args.workers,
        force      = args.force,
    )

    step_features(
        geo_out      = geo_out,
        run_tag      = args.run_tag,
        output_dir   = base_dir,
        features_out = features_out,
        workers      = args.workers,
        force        = args.force,
    )

    step_analysis(
        features_out = features_out,
        analysis_dir = analysis_dir,
        run_tag      = args.run_tag,
        workers      = args.workers,
        force        = args.force,
    )

    checkpoint_path = step_train(
        features_out         = features_out,
        run_tag              = args.run_tag,
        output_dir           = base_dir,
        exclude_features     = args.exclude_features,
        confidence_threshold = args.confidence_threshold,
        dropout              = args.dropout,
        device               = device,
        force                = args.force,
    )

    step_predict(
        features_out         = features_out,
        checkpoint_path      = checkpoint_path,
        output_csv           = output_csv,
        confidence_threshold = args.confidence_threshold,
        device               = device,
        force                = args.force,
    )

    figure_out = step_visualize(
        output_csv  = output_csv,
        ranges_file = ranges_out,
        dest_taxa   = args.dest_taxa,
        figures_dir = figures_dir,
        basemap_dir = args.basemap_dir,
        force       = args.force,
    )

    # ---- done --------------------------------------------------------------
    print()
    _sep()
    print("  LORE -- pipeline complete")
    _sep()
    print(f"  Final output     : {output_csv}")
    print(f"  Figure           : {figure_out}")
    print(f"  Checkpoint       : {checkpoint_path}")
    print(f"  Analysis         : {analysis_dir / 'analysis_report.txt'}")
    _sep()
    print()


if __name__ == "__main__":
    main()
