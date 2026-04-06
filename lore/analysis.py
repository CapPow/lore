"""
lore/analysis.py

Feature discriminability analysis for LORE (Latent Occurrence Resolution Engine).

Computes per-feature statistics that quantify how well each feature separates
the destination taxa using single-label, spatially confirmed records only.
Produces a human-readable report and a machine-readable CSV for use in
deciding which features to include or exclude from model training.

Statistics computed (per feature)
----------------------------------
    kw_h        Kruskal-Wallis H statistic — primary metric.
                Non-parametric one-way test; does not assume normality.
                Higher H = stronger between-group differences.
    kw_p        Kruskal-Wallis p-value. Values > 0.05 indicate the feature
                likely contributes no signal above chance. For large H
                statistics (H >> 100 at n > 10k), p underflows float64
                precision; values are clamped to 5e-324 in the CSV and
                displayed as p<2e-308 in the report.
    mi          Mutual information (bits) between feature and class label.
                Captures non-linear relationships. Computed via
                sklearn.feature_selection.mutual_info_classif.
    anova_f     ANOVA F-statistic. Parametric equivalent of KW-H; included
                for familiarity. Interpret with caution for non-normal
                distributions (most raster features).
    anova_p     ANOVA p-value.
    signal      Plain-language rating derived from kw_p and mi:
                  strong    kw_p < 0.001  and  mi > 0.05
                  moderate  kw_p < 0.05   and  mi > 0.01
                  weak      kw_p < 0.05   and  mi <= 0.01
                  flat      kw_p >= 0.05

Output files
------------
    <output_dir>/feature_stats.csv
        One row per feature, all statistics, signal rating.
        Column names match feat_* columns in features.parquet exactly —
        copy feature names directly into --exclude-features.

    <output_dir>/analysis_report.txt
        Human-readable summary grouped by feature block:
          - Coordinate block  (feat_lat, feat_lon)
          - Terrain block     (feat_elevation, feat_slope)
          - Bioclim block     (feat_bio1, feat_bio4, feat_bio12, feat_bio15)
          - Date block        (feat_sin_doy, feat_cos_doy)
          - Soil block        summary + per-class table
          - Name block        (feat_taxon_name_encoded)
        Flags weak/flat features explicitly with copy-paste exclude syntax.

Label filter
------------
Only records where suggested_names is a single taxon (no " | " separator,
not "out_of_range", not "excessive_uncertainty") are used. These are the
spatially confirmed, unambiguous records from geo.py.

Usage (library)
---------------
    from lore.analysis import run_analysis

    run_analysis(
        features="runs/peromyscus_split_2026/data/features.parquet",
        output_dir="runs/peromyscus_split_2026/analysis",
    )

Usage (CLI)
-----------
    python -m lore.analysis \\
        --features runs/peromyscus_split_2026/data/features.parquet \\
        [--output-dir runs/peromyscus_split_2026/analysis] \\
        [--workers 8]
"""

from __future__ import annotations

import argparse
import logging
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT     = Path(__file__).parent.parent
DEFAULT_OUT_DIR  = PROJECT_ROOT / "runs"
DEFAULT_WORKERS  = 8

LABEL_COL            = "suggested_names"
LABEL_OUT_OF_RANGE   = "out_of_range"
LABEL_EXCESSIVE      = "excessive_uncertainty"
LABEL_PARAPATRIC_SEP = " | "

# Feature blocks — order and grouping for the report
FEATURE_BLOCKS: dict[str, list[str]] = {
    "Coordinate":  ["feat_lat", "feat_lon"],
    "Terrain":     ["feat_elevation", "feat_slope"],
    "Bioclim":     ["feat_bio1", "feat_bio4", "feat_bio7", "feat_bio12", "feat_bio15"],
    "Date":        ["feat_sin_doy", "feat_cos_doy"],
    "Name":        ["feat_taxon_name_encoded"],
}
# Soil block is detected dynamically from column names

# Signal thresholds
STRONG_P   = 0.001
MODERATE_P = 0.05
STRONG_MI  = 0.05
MODERATE_MI = 0.01


# ---------------------------------------------------------------------------
# Label filtering
# ---------------------------------------------------------------------------

def _filter_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only single-label, spatially confirmed records.
    Excludes: out_of_range, excessive_uncertainty, parapatric (contains |).
    """
    mask = (
        ~df[LABEL_COL].str.contains(LABEL_PARAPATRIC_SEP, regex=False, na=False)
        & ~df[LABEL_COL].isin([LABEL_OUT_OF_RANGE, LABEL_EXCESSIVE])
        & df[LABEL_COL].notna()
    )
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Per-feature statistics
# ---------------------------------------------------------------------------

def _signal_rating(kw_p: float, mi: float) -> str:
    if kw_p < STRONG_P and mi > STRONG_MI:
        return "strong"
    if kw_p < MODERATE_P and mi > MODERATE_MI:
        return "moderate"
    if kw_p < MODERATE_P:
        return "weak"
    return "flat"


def _compute_feature_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    workers: int,
) -> pd.DataFrame:
    """
    Compute KW-H, KW-p, MI, ANOVA-F, ANOVA-p for each feature column.

    Parameters
    ----------
    df           : DataFrame containing feat_* columns, filtered to labeled records
    feature_cols : list of numeric feature column names to analyse
    labels       : integer-encoded class labels, length == len(df)
    workers      : n_jobs for mutual_info_classif

    Returns
    -------
    DataFrame with columns:
        feature, kw_h, kw_p, mi, anova_f, anova_p, signal
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)

    # Drop rows where ANY feature is NaN for the MI/ANOVA batch calls
    # (KW handles NaN per-feature below)
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid    = X[valid_mask]
    y_valid    = labels[valid_mask]

    logger.info(
        "  Computing mutual information on %d records (%d workers)...",
        int(valid_mask.sum()), workers,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_scores = mutual_info_classif(
            X_valid, y_valid,
            discrete_features=False,
            n_neighbors=3,
            random_state=42,
            n_jobs=workers,
        )

    logger.info("  Computing KW and ANOVA per feature...")
    rows = []
    unique_classes = np.unique(y_valid)

    for i, col in enumerate(tqdm(feature_cols, desc="feature stats", unit="feat")):
        x_all = X[:, i]
        # per-feature valid mask (NaN in this feature only)
        fmask  = ~np.isnan(x_all)
        x_feat = x_all[fmask]
        y_feat = labels[fmask]

        if len(x_feat) < 10 or len(np.unique(y_feat)) < 2:
            rows.append({
                "feature": col, "kw_h": np.nan, "kw_p": np.nan,
                "mi": mi_scores[i], "anova_f": np.nan, "anova_p": np.nan,
                "signal": "flat",
            })
            continue

        # Kruskal-Wallis
        groups = [x_feat[y_feat == c] for c in unique_classes if (y_feat == c).sum() > 0]
        groups = [g for g in groups if len(g) > 0]
        try:
            kw_h, kw_p = stats.kruskal(*groups)
        except Exception:
            kw_h, kw_p = np.nan, np.nan

        # ANOVA F
        try:
            anova_f, anova_p = stats.f_oneway(*groups)
        except Exception:
            anova_f, anova_p = np.nan, np.nan

        rows.append({
            "feature": col,
            "kw_h":    round(float(kw_h),    4) if not np.isnan(kw_h)    else np.nan,
            "kw_p":    float(kw_p)            if not np.isnan(kw_p)    else np.nan,
            "mi":      round(float(mi_scores[i]), 6),
            "anova_f": round(float(anova_f),  4) if not np.isnan(anova_f) else np.nan,
            "anova_p": float(anova_p)          if not np.isnan(anova_p) else np.nan,
            "signal":  _signal_rating(
                float(kw_p) if not np.isnan(kw_p) else 1.0,
                float(mi_scores[i]),
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt_row(feat: str, row: pd.Series, width: int = 42) -> str:
    """Format one feature row for the report."""
    name = feat.replace("feat_", "")
    _p  = row.kw_p
    _ps = "p<2e-308 " if _p == 0.0 else f"p={_p:.2e}"
    kw  = f"KW-H={row.kw_h:>10.2f}  {_ps}" if pd.notna(row.kw_h) else "KW-H=       N/A"

    mi   = f"MI={row.mi:.4f}"
    sig  = f"[{row.signal.upper()}]"
    return f"  {name:<{width}}  {kw}  {mi}  {sig}"


def _compute_soil_block_stats(
    df: pd.DataFrame,
    soil_cols: list[str],
    labels: np.ndarray,
) -> dict:
    """
    Compute block-level statistics for the full 123-dim soil vector.

    Returns a dict with:
        summed_mi       sum of per-class MI scores
        manova_f        Pillai trace approximation F-statistic
        manova_p        corresponding p-value
        n_records       records used (complete cases only)

    MANOVA implementation: Pillai's trace via scipy, computed on a
    random subsample of at most 5000 records per class for tractability.
    Full soil matrix at 127k records × 123 dims is computationally heavy;
    the subsample is sufficient for block-level signal detection.
    """
    # summed MI — fast, use pre-computed per-feature scores from stats_df
    # (passed in separately; computed here from raw data for independence)
    X_soil = df[soil_cols].to_numpy(dtype=np.float32)
    valid  = ~np.isnan(X_soil).any(axis=1)
    X_v    = X_soil[valid]
    y_v    = labels[valid]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_soil = mutual_info_classif(
            X_v, y_v,
            discrete_features=False,
            n_neighbors=3,
            random_state=42,
        )
    summed_mi = float(mi_soil.sum())

    # MANOVA via Pillai's trace — subsample for tractability
    rng          = np.random.default_rng(42)
    max_per_class = 5000
    idx_list = []
    for c in np.unique(y_v):
        c_idx = np.where(y_v == c)[0]
        if len(c_idx) > max_per_class:
            c_idx = rng.choice(c_idx, max_per_class, replace=False)
        idx_list.append(c_idx)
    idx    = np.concatenate(idx_list)
    X_sub  = X_v[idx]
    y_sub  = y_v[idx]

    try:
        classes     = np.unique(y_sub)
        grand_mean  = X_sub.mean(axis=0)
        n_total     = len(X_sub)
        p           = X_sub.shape[1]
        k           = len(classes)

        # Between-group scatter (H) and within-group scatter (E)
        H = np.zeros((p, p), dtype=np.float64)
        E = np.zeros((p, p), dtype=np.float64)
        for c in classes:
            mask    = y_sub == c
            n_c     = mask.sum()
            mean_c  = X_sub[mask].mean(axis=0)
            diff    = (mean_c - grand_mean).reshape(-1, 1)
            H      += n_c * (diff @ diff.T)
            Xc      = X_sub[mask] - mean_c
            E      += Xc.T @ Xc

        # Pillai's trace = trace(H(H+E)^-1)
        HplusE     = H + E
        HplusE_inv = np.linalg.pinv(HplusE)
        pillai     = float(np.trace(H @ HplusE_inv))

        # F approximation for Pillai's trace
        s   = min(k - 1, p)
        m   = (abs(p - (k - 1)) - 1) / 2
        nn  = (n_total - k - p - 1) / 2
        f_stat  = ((2 * nn + s + 1) / (2 * m + s + 1)) * (pillai / (s - pillai))
        df1     = s * (2 * m + s + 1)
        df2     = s * (2 * nn + s + 1)
        manova_p = float(1 - stats.f.cdf(f_stat, df1, df2))
        manova_f = float(f_stat)
    except Exception as exc:
        logger.warning("MANOVA computation failed: %s", exc)
        manova_f = np.nan
        manova_p = np.nan

    return {
        "summed_mi": round(summed_mi, 4),
        "manova_f":  round(manova_f, 4) if not np.isnan(manova_f) else np.nan,
        "manova_p":  manova_p,
        "n_records": int(valid.sum()),
    }


def _write_report(
    stats_df: pd.DataFrame,
    output_path: Path,
    n_labeled: int,
    n_total: int,
    n_classes: int,
    run_tag: str,
    soil_block_stats: dict | None = None,
) -> None:
    """Write the human-readable analysis report."""

    soil_cols = [f for f in stats_df["feature"] if "soil" in f]
    stats_idx = stats_df.set_index("feature")

    lines: list[str] = []
    sep = "=" * 70

    lines += [
        sep,
        "  LORE — Feature Discriminability Analysis Report",
        sep,
        f"  Run tag        : {run_tag}",
        f"  Total records  : {n_total:,}",
        f"  Labeled records: {n_labeled:,}  ({100*n_labeled/n_total:.1f}%  single-label only)",
        f"  Classes        : {n_classes}",
        f"  Features       : {len(stats_df)}  (123 soil + 10 scalar + 1 name)",
        "",
        "  Statistics key:",
        "    KW-H   Kruskal-Wallis H  (primary — non-parametric, no normality assumption)",
        "    p      p-value           (> 0.05 → likely no signal)",
        "    MI     Mutual information (bits, captures non-linear relationships)",
        "    ANOVA  F-statistic        (parametric reference — interpret cautiously)",
        "",
        "  Signal ratings:",
        "    STRONG    kw_p < 0.001  and  MI > 0.05",
        "    MODERATE  kw_p < 0.05   and  MI > 0.01",
        "    WEAK      kw_p < 0.05   and  MI <= 0.01",
        "    FLAT      kw_p >= 0.05  (candidate for exclusion)",
        sep,
        "",
    ]

    # ---- scalar blocks -----------------------------------------------------
    for block_name, block_cols in FEATURE_BLOCKS.items():
        present = [c for c in block_cols if c in stats_idx.index]
        if not present:
            continue
        lines.append(f"[ {block_name} block ]")
        for col in present:
            lines.append(_fmt_row(col, stats_idx.loc[col]))
        lines.append("")

    # ---- soil block — summary + flagged ------------------------------------
    if soil_cols:
        soil_stats = stats_idx.loc[soil_cols]
        signal_counts = soil_stats["signal"].value_counts()

        lines.append("[ Soil block  —  123 classes ]")

        # Block-level multivariate stats
        if soil_block_stats:
            smi  = soil_block_stats["summed_mi"]
            mf   = soil_block_stats["manova_f"]
            mp   = soil_block_stats["manova_p"]
            nr   = soil_block_stats["n_records"]
            mf_s = f"{mf:.4f}" if not (isinstance(mf, float) and np.isnan(mf)) else "N/A"
            mp_s = f"{mp:.2e}" if not (isinstance(mp, float) and np.isnan(mp)) else "N/A"
            lines += [
                f"  Block-level statistics (full 123-dim soil vector, n={nr:,}):",
                f"    Summed MI (total soil block information) : {smi:.4f} bits",
                f"    MANOVA Pillai F-approx                  : {mf_s}",
                f"    MANOVA p-value                          : {mp_s}",
                "",
            ]

        lines.append(f"  Per-class signal distribution:")
        for sig in ["strong", "moderate", "weak", "flat"]:
            count = signal_counts.get(sig, 0)
            lines.append(f"    {sig.upper():<10}  {count:>4}  classes")
        lines.append("")

        lines.append(f"  Top 10 highest MI soil classes:")
        top10 = soil_stats.nlargest(10, "mi")
        for feat, row in top10.iterrows():
            lines.append(_fmt_row(feat, row, width=50))
        lines.append("")

        flat_soil = soil_stats[soil_stats["signal"] == "flat"]
        weak_soil = soil_stats[soil_stats["signal"] == "weak"]

        lines.append(f"  Flat soil classes ({len(flat_soil)})  — candidates for exclusion:")
        if len(flat_soil) == 0:
            lines.append("    None.")
        else:
            for feat, row in flat_soil.iterrows():
                lines.append(_fmt_row(feat, row, width=50))
        lines.append("")

        lines.append(f"  Weak soil classes ({len(weak_soil)}):")
        if len(weak_soil) == 0:
            lines.append("    None.")
        else:
            for feat, row in weak_soil.iterrows():
                lines.append(_fmt_row(feat, row, width=50))
        lines += [
            "",
            "  NOTE ON WEAK SOIL CLASSES:",
            "  Low individual MI for a soil class does not mean it should be",
            "  excluded. Many weak classes (e.g. tropical/arctic soils) have",
            "  near-zero probability across all records — their signal is",
            "  geographic absence, not noise. Excluding them removes information",
            "  and creates holes in the feature space for non-target taxa.",
            "  Recommendation: retain all soil classes unless MANOVA p > 0.05",
            "  (i.e. the full soil vector contributes no block-level signal).",
            "",
        ]

    # ---- flat / weak scalar features ---------------------------------------
    scalar_features = [
        c for block in FEATURE_BLOCKS.values() for c in block
        if c in stats_idx.index
    ]
    flat_scalar = [
        c for c in scalar_features
        if stats_idx.loc[c, "signal"] in ("flat", "weak")
    ]
    lines.append("[ Scalar features flagged as WEAK or FLAT ]")
    if not flat_scalar:
        lines.append("  None — all scalar features show at least weak signal.")
    else:
        for col in flat_scalar:
            lines.append(_fmt_row(col, stats_idx.loc[col]))
    lines.append("")

    # ---- exclude-features hint ---------------------------------------------
    all_flat = stats_df[stats_df["signal"] == "flat"]["feature"].tolist()
    lines += [
        sep,
        "  FEATURE EXCLUSION GUIDANCE",
        sep,
        "  Features rated FLAT across KW and MI are candidates for exclusion.",
        "  Review the stats CSV for full per-feature detail before deciding.",
        "  To exclude features, pass to model training:",
        "",
    ]
    if all_flat:
        exclude_str = " \\\n    ".join(all_flat)
        lines.append(
            "    python -m lore.model ... \\\n"
            f"      --exclude-features \\\n"
            f"    {exclude_str}"
        )
    else:
        lines.append("  No flat features detected — no exclusions recommended.")
    lines += ["", sep, "  Full per-feature statistics: feature_stats.csv", sep]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def run_analysis(
    features: str | Path,
    output_dir: str | Path = DEFAULT_OUT_DIR,
    run_tag: str = "",
    workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    """
    Run feature discriminability analysis on a LORE features parquet.

    Parameters
    ----------
    features   : path to features.parquet (output of lore.features)
    output_dir : directory for output files (created if needed)
    run_tag    : identifier string included in the report header
    workers    : n_jobs for mutual_info_classif

    Returns
    -------
    pd.DataFrame of per-feature statistics (also written to feature_stats.csv)

    Outputs
    -------
    <output_dir>/feature_stats.csv
    <output_dir>/analysis_report.txt
    """
    features   = Path(features)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load --------------------------------------------------------------
    logger.info("Loading features from %s", features)
    df = pd.read_parquet(features)
    n_total = len(df)
    logger.info("  %d total records loaded.", n_total)

    # ---- filter to labeled records -----------------------------------------
    labeled = _filter_labeled(df)
    n_labeled = len(labeled)
    logger.info(
        "  %d single-label records retained for analysis (%.1f%%).",
        n_labeled, 100 * n_labeled / n_total,
    )

    if n_labeled < 100:
        raise ValueError(
            f"Too few labeled records for analysis ({n_labeled}). "
            f"Check that suggested_names is populated in the features parquet."
        )

    # ---- encode labels -----------------------------------------------------
    le = LabelEncoder()
    y  = le.fit_transform(labeled[LABEL_COL].to_numpy())
    n_classes = len(le.classes_)
    logger.info("  %d classes: %s", n_classes, list(le.classes_))

    # ---- identify numeric feature columns ----------------------------------
    all_feat_cols = [
        c for c in labeled.columns
        if c.startswith("feat_")
        and c not in {"feat_taxon_name", "feat_taxon_name_encoded", "feat_has_nodata"}
    ]

    # Drop columns that are entirely NaN in the labeled subset
    non_null = [
        c for c in all_feat_cols
        if labeled[c].notna().any()
    ]
    dropped_null = set(all_feat_cols) - set(non_null)
    if dropped_null:
        logger.warning(
            "  %d feature columns are entirely NaN in labeled records — skipped: %s",
            len(dropped_null), sorted(dropped_null),
        )

    logger.info("  Analysing %d feature columns...", len(non_null))

    # ---- compute stats -----------------------------------------------------
    stats_df = _compute_feature_stats(labeled, non_null, y, workers)

    # ---- soil block multivariate stats -------------------------------------
    soil_cols = [c for c in non_null if "soil" in c]
    soil_block_stats = None
    if soil_cols:
        logger.info(
            "  Computing soil block MANOVA on %d classes × %d soil features...",
            n_classes, len(soil_cols),
        )
        soil_block_stats = _compute_soil_block_stats(labeled, soil_cols, y)
        logger.info(
            "  Soil block — summed MI=%.4f  MANOVA F=%.4f  p=%.2e",
            soil_block_stats["summed_mi"],
            soil_block_stats["manova_f"] if not np.isnan(soil_block_stats["manova_f"]) else 0,
            soil_block_stats["manova_p"] if not np.isnan(soil_block_stats["manova_p"]) else 1,
        )

    # ---- write CSV ---------------------------------------------------------
    csv_path = output_dir / "feature_stats.csv"
    
    csv_out = stats_df.copy()
    csv_out["kw_p"]   = csv_out["kw_p"].apply(lambda v: 5e-324 if v == 0.0 else v)
    csv_out["anova_p"] = csv_out["anova_p"].apply(lambda v: 5e-324 if v == 0.0 else v)
    csv_out.to_csv(csv_path, index=False, float_format="%.6e")
    
    logger.info("Feature stats written to %s", csv_path)

    # ---- write report ------------------------------------------------------
    report_path = output_dir / "analysis_report.txt"
    _write_report(
        stats_df=stats_df,
        output_path=report_path,
        n_labeled=n_labeled,
        n_total=n_total,
        n_classes=n_classes,
        run_tag=run_tag or features.parent.name,
        soil_block_stats=soil_block_stats,
    )
    logger.info("Analysis report written to %s", report_path)

    # ---- print summary to stdout -------------------------------------------
    signal_counts = stats_df["signal"].value_counts()
    flat_features = stats_df[stats_df["signal"] == "flat"]["feature"].tolist()

    print()
    print("=" * 55)
    print("  LORE — feature analysis summary")
    print("=" * 55)
    print(f"  Labeled records  : {n_labeled:,}  of  {n_total:,}")
    print(f"  Classes          : {n_classes}")
    print(f"  Features analysed: {len(stats_df)}")
    print()
    print("  Signal distribution:")
    for sig in ["strong", "moderate", "weak", "flat"]:
        count = signal_counts.get(sig, 0)
        print(f"    {sig.upper():<10}  {count:>4}")
    print()
    if flat_features:
        print(f"  {len(flat_features)} FLAT features (exclusion candidates):")
        for f in flat_features:
            print(f"    {f}")
    else:
        print("  No flat features detected.")
    print()
    print(f"  Stats CSV  : {csv_path}")
    print(f"  Report     : {report_path}")
    print("=" * 55)

    return stats_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Feature discriminability analysis for a LORE features parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features", type=Path, required=True,
                   help="Path to features.parquet.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Directory for output files.")
    p.add_argument("--run-tag", default="",
                   help="Identifier string for the report header.")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="Parallel workers for mutual information computation.")
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
    print("  LORE — feature discriminability analysis")
    print("=" * 55)
    print(f"  Features   : {args.features}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Workers    : {args.workers}")
    print("=" * 55)

    run_analysis(
        features=args.features,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
