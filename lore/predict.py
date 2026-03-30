"""
lore/predict.py

Inference and final output production for LORE (Latent Occurrence Resolution Engine).

Loads a trained checkpoint (output of lore/model.py) and a features parquet
(output of lore/features.py), runs inference on ambiguous records, and produces
a final CSV with one row per occurrence and a clean resolved taxon assignment.

Disambiguation logic
--------------------
Records are processed in four mutually exclusive categories:

    geo             Single-label, spatially confirmed by geo.py.
                    final_taxon = suggested_names. ML not run (would be
                    circular -- these records include training data).

    ml              Ambiguous (parapatric or out_of_range), successfully
                    disambiguated by the model above confidence threshold.
                    final_taxon = ml_prediction.

    ml_low_confidence
                    Ambiguous, model prediction below confidence threshold
                    (only applies when --confidence-threshold is set).
                    final_taxon = ml_prediction (flagged -- treat with caution).

    excluded        Ambiguous but excluded from ML inference due to:
                    - excessive_uncertainty label from geo.py
                    - NaN in one or more active feature columns
                    final_taxon = null.

Output columns
--------------
All original Darwin Core columns are preserved, plus:

    final_taxon             Clean resolved taxon name (str or null).
                            The primary output column for downstream use.
    disambiguation_method   geo / ml / ml_low_confidence / excluded
    suggested_names         Original geo.py label (preserved unchanged)
    ml_prediction           Top model prediction (null for geo records)
    ml_confidence           Softmax max probability (null for geo records)
    ml_candidate_match      Bool -- for parapatric records, was ml_prediction
                            one of the pipe-delimited candidates? (null otherwise)
    feat_has_nodata         Raster coverage flag from features.py

Usage (library)
---------------
    from lore.predict import predict

    predict(
        features="runs/peromyscus_split_2026/features.parquet",
        checkpoint="runs/peromyscus_split_2026/cache/model/checkpoint.pt",
        output="runs/peromyscus_split_2026/disambiguated.csv",
    )

Usage (CLI)
-----------
    python -m lore.predict \\
        --features   runs/peromyscus_split_2026/features.parquet \\
        --checkpoint runs/peromyscus_split_2026/cache/model/checkpoint.pt \\
        --output     runs/peromyscus_split_2026/disambiguated.csv \\
        [--confidence-threshold 0.8] \\
        [--impute-inference] \\
        [--batch-size 1024]
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lore.model import (
    LoreNet,
    NAME_FEATURE,
    LABEL_COL,
    LABEL_OUT_OF_RANGE,
    LABEL_EXCESSIVE,
    LABEL_PARAPATRIC_SEP,
    _filter_single_label,
    _filter_parapatric,
    _recompute_nodata_mask,
    _build_tensors,
    build_model_from_checkpoint,
    load_checkpoint,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _apply_inference_imputation(
    df: pd.DataFrame,
    feature_cols: list[str],
    class_means: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Apply class-conditional mean imputation at inference time using means
    serialized from the training split.

    For inference records the class is unknown, so the grand mean across all
    training classes is used as a fallback for each feature.

    Parameters
    ----------
    df           : DataFrame of records to impute
    feature_cols : active feature columns
    class_means  : {class_int: {feature: mean}} from checkpoint

    Returns
    -------
    Copy of df with NaN values filled.
    """
    if not class_means:
        warnings.warn(
            "No class-conditional means found in checkpoint. "
            "Falling back to column grand mean for imputation.",
            stacklevel=2,
        )
        return df.fillna(df[feature_cols].mean())

    df = df.copy()
    grand_means = {
        col: float(np.mean([class_means[c][col] for c in class_means]))
        for col in feature_cols
    }
    for col in feature_cols:
        null_mask = df[col].isna()
        if null_mask.any():
            df.loc[null_mask, col] = grand_means[col]
    return df


def _run_inference(
    model: LoreNet,
    df: pd.DataFrame,
    numeric_cols: list[str],
    soil_cols: list[str],
    date_cols: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run forward pass on df. Returns (pred_indices, confidences).

    pred_indices : int array of argmax class indices
    confidences  : float32 array of softmax max probabilities
    """
    tensors = _build_tensors(df, numeric_cols, soil_cols, date_cols, None, device)
    dataset = TensorDataset(*tensors)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_num, x_soil, x_date, x_name = [t.to(device) for t in batch]
            logits = model(x_num, x_soil, x_date, x_name)
            all_probs.append(torch.softmax(logits, dim=1).cpu())

    probs       = torch.cat(all_probs, dim=0).numpy()
    pred_idx    = probs.argmax(axis=1).astype(np.int32)
    confidences = probs.max(axis=1).astype(np.float32)
    return pred_idx, confidences


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

def predict(
    features: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    confidence_threshold: float | None = None,
    impute_inference: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device_str: str = "auto",
) -> pd.DataFrame:
    """
    Run inference and produce final disambiguation output.

    Parameters
    ----------
    features             : path to features.parquet
    checkpoint           : path to checkpoint.pt from lore/model.py
    output               : path for output CSV
    confidence_threshold : softmax probability below which predictions are
                           flagged as ml_low_confidence. Default None (uses
                           value from checkpoint, or no threshold if unset).
                           Confidence scores are always written regardless.
    impute_inference     : apply serialized class-conditional means to
                           NaN features at inference time instead of
                           excluding records. Default False.
    batch_size           : inference batch size
    device_str           : 'auto', 'cpu', 'cuda', or 'mps'

    Returns
    -------
    pandas.DataFrame of the full output (also written to output CSV).
    """
    features   = Path(features)
    checkpoint = Path(checkpoint)
    output     = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ---- device ------------------------------------------------------------
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # ---- load checkpoint ---------------------------------------------------
    logger.info("Loading checkpoint from %s", checkpoint)
    ckpt = load_checkpoint(checkpoint)

    class_names  = ckpt["class_names"]
    numeric_cols = ckpt["numeric_cols"]
    soil_cols    = ckpt["soil_cols"]
    date_cols    = ckpt["date_cols"]
    class_means  = ckpt.get("class_means", {})
    ckpt_thresh  = ckpt.get("confidence_threshold")
    active_cols  = numeric_cols + soil_cols + date_cols

    # Honour checkpoint threshold if caller did not override
    if confidence_threshold is None:
        confidence_threshold = ckpt_thresh

    logger.info("  Classes: %s", class_names)
    logger.info(
        "  Confidence threshold: %s",
        confidence_threshold if confidence_threshold is not None else "none",
    )

    # ---- load features -----------------------------------------------------
    logger.info("Loading features from %s", features)
    df      = pd.read_parquet(features)
    n_total = len(df)
    logger.info("  %d total records.", n_total)

    # ---- build model -------------------------------------------------------
    # build_model_from_checkpoint reads from checkpoint["architecture"],
    # which is the single authoritative source for LoreNet constructor args.
    model = build_model_from_checkpoint(ckpt, device)
    logger.info(
        "  Model loaded (%d params).",
        sum(p.numel() for p in model.parameters()),
    )

    # ---- partition records -------------------------------------------------
    geo_mask = (
        ~df[LABEL_COL].str.contains(LABEL_PARAPATRIC_SEP, regex=False, na=False)
        & ~df[LABEL_COL].isin([LABEL_OUT_OF_RANGE, LABEL_EXCESSIVE])
        & df[LABEL_COL].notna()
    )
    excessive_mask = df[LABEL_COL] == LABEL_EXCESSIVE
    ambiguous_mask = ~geo_mask & ~excessive_mask

    df_geo       = df[geo_mask].copy()
    df_excessive = df[excessive_mask].copy()
    df_ambiguous = df[ambiguous_mask].copy()

    logger.info(
        "  Geo-resolved: %d  |  Ambiguous: %d  |  Excessive uncertainty: %d",
        len(df_geo), len(df_ambiguous), len(df_excessive),
    )

    # ---- nodata handling for ambiguous records -----------------------------
    nodata_mask = _recompute_nodata_mask(
        df_ambiguous, numeric_cols, soil_cols, date_cols
    )
    n_nodata = int(nodata_mask.sum())

    if n_nodata:
        logger.info(
            "  %d ambiguous records have NaN in active features.", n_nodata
        )
        if impute_inference:
            logger.info("  Applying class-conditional mean imputation...")
            df_ambiguous = _apply_inference_imputation(
                df_ambiguous, active_cols, class_means
            )
            # Recompute mask after imputation -- should be all-False if
            # imputation covered all NaN columns
            nodata_mask = _recompute_nodata_mask(
                df_ambiguous, numeric_cols, soil_cols, date_cols
            )
        else:
            logger.info(
                "  %d records will be excluded (nodata). "
                "Pass --impute-inference to attempt disambiguation.",
                n_nodata,
            )

    df_excluded_nodata = df_ambiguous[nodata_mask].copy()
    df_to_infer        = df_ambiguous[~nodata_mask].copy()

    logger.info(
        "  Running ML inference on %d ambiguous records...", len(df_to_infer)
    )

    # ---- inference ---------------------------------------------------------
    results: dict[int, dict] = {}  # index -> result dict

    # Geo records: final_taxon is the geo label, no ML run
    for idx in df_geo.index:
        results[idx] = {
            "final_taxon":           df_geo.loc[idx, LABEL_COL],
            "disambiguation_method": "geo",
            "ml_prediction":         None,
            "ml_confidence":         None,
            "ml_candidate_match":    None,
        }

    # Excessive uncertainty: excluded entirely
    for idx in df_excessive.index:
        results[idx] = {
            "final_taxon":           None,
            "disambiguation_method": "excluded",
            "ml_prediction":         None,
            "ml_confidence":         None,
            "ml_candidate_match":    None,
        }

    # Ambiguous with nodata: excluded
    for idx in df_excluded_nodata.index:
        results[idx] = {
            "final_taxon":           None,
            "disambiguation_method": "excluded",
            "ml_prediction":         None,
            "ml_confidence":         None,
            "ml_candidate_match":    None,
        }

    # ML inference on remaining ambiguous records
    if len(df_to_infer) > 0:
        pred_idx, confidences = _run_inference(
            model, df_to_infer, numeric_cols, soil_cols, date_cols,
            device, batch_size,
        )
        pred_names = [class_names[i] for i in pred_idx]

        for i, (idx, row) in enumerate(df_to_infer.iterrows()):
            pred       = pred_names[i]
            conf       = float(confidences[i])
            candidates = row[LABEL_COL]
            is_para    = LABEL_PARAPATRIC_SEP in str(candidates)

            # candidate_match is only meaningful for parapatric records where
            # the set of plausible taxa is known; out_of_range has no candidates
            if is_para:
                candidate_set = set(candidates.split(LABEL_PARAPATRIC_SEP))
                match = pred in candidate_set
            else:
                match = None

            if confidence_threshold is not None and conf < confidence_threshold:
                method = "ml_low_confidence"
            else:
                method = "ml"

            results[idx] = {
                "final_taxon":           pred,
                "disambiguation_method": method,
                "ml_prediction":         pred,
                "ml_confidence":         round(conf, 6),
                "ml_candidate_match":    match,
            }

    # ---- assemble output ---------------------------------------------------
    result_df = pd.DataFrame.from_dict(results, orient="index")
    result_df.index.name = df.index.name

    out_df = df.join(result_df, how="left")

    # DwC columns first, then disambiguation columns
    disambig_cols = [
        "final_taxon",
        "disambiguation_method",
        "suggested_names",
        "ml_prediction",
        "ml_confidence",
        "ml_candidate_match",
        "feat_has_nodata",
    ]
    other_cols = [c for c in out_df.columns if c not in disambig_cols]
    out_df = out_df[other_cols + disambig_cols]

    # Drop feat_* and geometry -- preserved in features.parquet and redundant
    # with decimalLatitude/decimalLongitude in the output CSV
    drop_cols = [c for c in out_df.columns if c.startswith("feat_")] + ["geometry"]
    out_df = out_df.drop(columns=[c for c in drop_cols if c in out_df.columns])

    # ---- write output ------------------------------------------------------
    out_df.to_csv(output, index=False)
    logger.info("Output written to %s", output)

    # ---- summary -----------------------------------------------------------
    method_counts = out_df["disambiguation_method"].value_counts()
    n_resolved    = int((out_df["disambiguation_method"] != "excluded").sum())
    n_excluded    = int((out_df["disambiguation_method"] == "excluded").sum())
    n_ml          = int(method_counts.get("ml", 0))
    n_ml_low      = int(method_counts.get("ml_low_confidence", 0))
    n_geo         = int(method_counts.get("geo", 0))

    para_mask    = out_df["ml_candidate_match"].notna()
    n_para       = int(para_mask.sum())
    n_para_match = int(out_df.loc[para_mask, "ml_candidate_match"].sum())

    print()
    print("=" * 55)
    print("  LORE -- disambiguation complete")
    print("=" * 55)
    print(f"  Total records       : {n_total:,}")
    print(f"  Geo-resolved        : {n_geo:,}  ({100*n_geo/n_total:.1f}%)")
    print(f"  ML-resolved         : {n_ml:,}  ({100*n_ml/n_total:.1f}%)")
    if n_ml_low:
        print(f"  ML low-confidence   : {n_ml_low:,}  ({100*n_ml_low/n_total:.1f}%)")
    print(f"  Excluded            : {n_excluded:,}  ({100*n_excluded/n_total:.1f}%)")
    print(f"  Total resolved      : {n_resolved:,}  ({100*n_resolved/n_total:.1f}%)")
    if n_para > 0:
        print(
            f"  Para candidate match: {n_para_match:,} / {n_para:,}"
            f"  ({100*n_para_match/n_para:.1f}%)"
        )
    if confidence_threshold is not None:
        print(f"  Conf. threshold     : {confidence_threshold}")
    print()

    print("  Final taxon distribution:")
    taxon_counts = out_df["final_taxon"].value_counts(dropna=False)
    for taxon, count in taxon_counts.items():
        label = taxon if pd.notna(taxon) else "(excluded/null)"
        print(f"    {label:<45}  {count:>8,}  ({100*count/n_total:.1f}%)")

    print()
    print(f"  Output: {output}")
    print("=" * 55)

    return out_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run LORE inference and produce final disambiguation output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features", type=Path, required=True,
                   help="Path to features.parquet.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to checkpoint.pt from lore/model.py.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--confidence-threshold", type=float, default=None,
                   help="Softmax probability below which predictions are flagged "
                        "ml_low_confidence. Default: use value from checkpoint "
                        "(or none if not set). Confidence scores always written.")
    p.add_argument("--impute-inference", action="store_true", default=False,
                   help="Apply serialized class-conditional means to NaN features "
                        "at inference time instead of excluding records.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="Inference batch size.")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
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
    print("  LORE -- inference")
    print("=" * 55)
    print(f"  Features    : {args.features}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Output      : {args.output}")
    print(f"  Device      : {args.device}")
    print(f"  Conf. thresh: {args.confidence_threshold or 'none'}")
    print("=" * 55)

    predict(
        features             = args.features,
        checkpoint           = args.checkpoint,
        output               = args.output,
        confidence_threshold = args.confidence_threshold,
        impute_inference     = args.impute_inference,
        batch_size           = args.batch_size,
        device_str           = args.device,
    )


if __name__ == "__main__":
    main()
