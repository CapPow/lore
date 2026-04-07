"""
lore/model.py

Multi-input neural network for taxonomic disambiguation of GBIF occurrence
records following a taxonomic split.

Trains on single-label, spatially confirmed records from features.parquet
(output of lore/features.py) and evaluates on both clean held-out records
and parapatric records (where the model's prediction is checked against the
set of known candidate taxa).

Network architecture
--------------------
Five parallel encoder streams, each a stack of Linear -> ELU -> Dropout
layers, whose outputs are concatenated and passed through a shared decoder
to a softmax output head.

    numeric stream   feat_lat, feat_lon (optional), feat_elevation,
                     feat_slope, feat_bio1/4/7/12/15
    soil stream      feat_soil_* (123-dim probability vector)
                     Uses a dedicated depth (soil_encoder_depth, default 4)
                     to give the high-dimensional soil block sufficient
                     capacity before the merge.
    land cover stream feat_lc_* (12-dim EarthEnv consensus cover fractions)
                     Separate stream: habitat structure is ecologically
                     independent of climate and soil type.
    name stream      feat_verbatim_name_encoded (embedding lookup)
                     Disabled automatically when only one source taxon
                     is present -- a constant embedding carries no signal.
    date stream      feat_sin_doy, feat_cos_doy

Rationale for separate streams: feature blocks have very different
statistical properties (MI ranges from ~0.001 to ~0.87). Shared early
layers allow high-MI features to dominate gradients. Separate encoders
let each block develop its own representation before the merge.

Class imbalance
---------------
Inverse-frequency class weights are computed from the training split and
passed to CrossEntropyLoss. A warning is raised if max/min class ratio
exceeds IMBALANCE_WARN_RATIO (default 100).

Missing data
------------
Training:   class-conditional mean imputation. Per-feature means computed
            separately per class on the training split, then serialized to
            the checkpoint. Records missing features in excluded columns
            are not penalized.
Inference:  records with NaN in any included feature are dropped by default
            and flagged as 'excluded' in the output. Pass --impute-inference
            to apply serialized class-conditional means instead.

Feature exclusion
-----------------
Pass --exclude-features to drop any feat_* columns before training.
The nodata mask is recomputed after exclusion so records are only dropped
for NaN in features that are actually used.
Recommended diagnostic runs:
    1. Full feature set (baseline)
    2. --exclude-features feat_lat feat_lon  (geography-free)
    3. --exclude-features feat_lat feat_lon + soil  (climate/date/name only)

Checkpoint format
-----------------
The checkpoint produced by train() is fully self-describing for inference.
All information needed to reconstruct LoreNet and run predict.py is
serialized into the checkpoint -- no external config file is required.

Key sections:
    architecture        LoreNet constructor arguments (n_numeric, n_soil,
                        n_vocab, n_classes, use_name_stream). Use
                        build_model_from_checkpoint() to reconstruct.
    hyperparameters     Training configuration (hidden_dim, dropout, lr,
                        encoder_depth, soil_encoder_depth, etc.). Stored
                        for reproducibility; not needed for inference.
    model_state_dict    Trained weights.
    numeric_cols        Ordered list of numeric feature names used.
    soil_cols           Ordered list of soil feature names used.
    lc_cols             Ordered list of land cover feature names used.
    date_cols           Ordered list of date feature names used.
    class_names         Ordered list of output class names.
    class_means         Per-class imputation means from training split.
    confidence_threshold Softmax threshold used during training (may be
                        overridden at inference time).

Outputs (written to <output-dir>/<run-tag>/cache/model/)
---------------------------------------------------------
    checkpoint.pt           Self-contained model checkpoint (see above).
    training_log.csv        Per-epoch loss and accuracy (train + val).
    training_summary.txt    Hyperparameters, class weights, final metrics,
                            parapatric resolution rate, feature list.
Dropout guidance
----------------
The optimal dropout rate is inversely correlated with dataset size. The
default (0.1) is appropriate for large datasets (~60k+ training records).
For smaller datasets, increasing dropout is advised:

    ~3,500 training records  ->  0.25
    ~60,000+ training records -> 0.10

Intermediate sizes should treat --dropout as the primary tuning knob.
This relationship was characterized using Peromyscus (~86k records, 6
classes) and Chaetodipus (~5k records, 3 classes) as bounding examples.
See docs/hyperparameter_sweep.md for full sweep results and methodology.

Usage (library)
---------------
    from lore.model import train

    results = train(
        features="runs/peromyscus_split_2026/features.parquet",
        run_tag="peromyscus_split_2026",
        output_dir="runs",
        exclude_features=["feat_lat", "feat_lon"],
    )

Usage (CLI)
-----------
    python -m lore.model \\
        --features    runs/peromyscus_split_2026/features.parquet \\
        --run-tag     peromyscus_split_2026 \\
        --output-dir  runs \\
        [--exclude-features feat_lat feat_lon] \\
        [--confidence-threshold 0.8] \\
        [--drop-nodata | --impute-nodata] \\
        [--soil-encoder-depth 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT       = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"

LABEL_COL            = "suggested_names"
LABEL_OUT_OF_RANGE   = "out_of_range"
LABEL_EXCESSIVE      = "excessive_uncertainty"
LABEL_PARAPATRIC_SEP = " | "

# Feature block definitions -- order matters for stream assembly
NUMERIC_FEATURES = [
    "feat_lat", "feat_lon",
    "feat_elevation", "feat_slope",
    "feat_bio1", "feat_bio4", "feat_bio7", "feat_bio12", "feat_bio15",
]
DATE_FEATURES = ["feat_sin_doy", "feat_cos_doy"]
NAME_FEATURE  = "feat_taxon_name_encoded"

# Class imbalance warning threshold (max_count / min_count)
IMBALANCE_WARN_RATIO = 100

# Training defaults
DEFAULT_VAL_FRAC           = 0.15
DEFAULT_TEST_FRAC          = 0.15
DEFAULT_BATCH_SIZE         = 256
DEFAULT_MAX_EPOCHS         = 500
DEFAULT_PATIENCE           = 50
DEFAULT_LR                 = 5e-4
DEFAULT_DROPOUT            = 0.1
DEFAULT_EMBED_DIM          = 32   # name embedding dimension
DEFAULT_SOIL_ENCODER_DEPTH = 4    # deeper than numeric/date encoders given 123-dim input
DEFAULT_LC_ENCODER_DEPTH   = 3    # shallower than soil given 12-dim input


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _filter_single_label(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only single-label, spatially confirmed records."""
    mask = (
        ~df[LABEL_COL].str.contains(LABEL_PARAPATRIC_SEP, regex=False, na=False)
        & ~df[LABEL_COL].isin([LABEL_OUT_OF_RANGE, LABEL_EXCESSIVE])
        & df[LABEL_COL].notna()
    )
    return df[mask].copy()


def _filter_parapatric(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only parapatric records (pipe-delimited suggested_names)."""
    mask = df[LABEL_COL].str.contains(LABEL_PARAPATRIC_SEP, regex=False, na=False)
    return df[mask].copy()


def _resolve_feature_cols(
    df: pd.DataFrame,
    exclude_features: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Resolve which feature columns to use given exclusions.

    Returns
    -------
    numeric_cols   : numeric stream feature names (subset of NUMERIC_FEATURES)
    soil_cols      : soil stream feature names
    lc_cols        : land cover stream feature names
    date_cols      : date stream feature names (subset of DATE_FEATURES)

    Note: NAME_FEATURE is handled separately via use_name_stream in train().
    """
    excluded = set(exclude_features or [])

    numeric_cols = [f for f in NUMERIC_FEATURES if f in df.columns and f not in excluded]
    date_cols    = [f for f in DATE_FEATURES    if f in df.columns and f not in excluded]
    soil_cols    = [
        c for c in df.columns
        if c.startswith("feat_soil_") and c not in excluded
    ]
    lc_cols = [
        c for c in df.columns
        if c.startswith("feat_lc_") and c not in excluded
    ]

    if not numeric_cols:
        warnings.warn("All numeric features excluded -- numeric stream will be empty.")
    if not soil_cols:
        warnings.warn("All soil features excluded -- soil stream will be empty.")
    if not lc_cols:
        warnings.warn("No land cover features found -- land cover stream will be empty.")
    if not date_cols:
        warnings.warn("All date features excluded -- date stream will be empty.")

    return numeric_cols, soil_cols, lc_cols, date_cols


def _recompute_nodata_mask(
    df: pd.DataFrame,
    numeric_cols: list[str],
    soil_cols: list[str],
    lc_cols: list[str],
    date_cols: list[str],
) -> pd.Series:
    """
    Recompute nodata mask using only the features that will actually be used.
    Records are only flagged if they are missing a feature in the active set.
    """
    active = numeric_cols + soil_cols + lc_cols + date_cols
    if not active:
        return pd.Series(False, index=df.index)
    return df[active].isna().any(axis=1)


def _compute_class_conditional_means(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    classes: np.ndarray,
) -> dict[int, dict[str, float]]:
    """
    Compute per-class, per-feature means on the training split for imputation.

    Returns
    -------
    dict: {class_int: {feature_name: mean_value}}
    Falls back to grand mean for features with no valid values in a class.
    """
    grand_means = {col: float(df[col].mean()) for col in feature_cols}
    means: dict[int, dict[str, float]] = {}
    for c in classes:
        mask   = labels == c
        subset = df[feature_cols][mask]
        means[int(c)] = {
            col: float(subset[col].mean()) if subset[col].notna().any()
            else grand_means[col]
            for col in feature_cols
        }
    return means


def _apply_imputation(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    class_means: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Apply class-conditional mean imputation in-place on a copy.
    For records where the class label is known (training).
    """
    df = df.copy()
    for i, (idx, row) in enumerate(df[feature_cols].iterrows()):
        c = int(labels[i])
        for col in feature_cols:
            if pd.isna(row[col]):
                df.at[idx, col] = class_means[c][col]
    return df


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

def _make_encoder(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int,
    dropout: float,
    activation: nn.Module,
) -> nn.Sequential:
    """
    Build a stream encoder: Linear -> activation -> Dropout x depth -> Linear.
    Returns an empty Sequential if in_dim == 0 (stream disabled or empty).
    """
    if in_dim == 0:
        return nn.Sequential()

    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        activation,
        nn.Dropout(dropout),
    ]
    current_dim = hidden_dim
    for _ in range(depth - 1):
        next_dim = max(out_dim, current_dim // 2)
        layers += [
            nn.Linear(current_dim, next_dim),
            activation,
            nn.Dropout(dropout),
        ]
        current_dim = next_dim

    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class LoreNet(nn.Module):
    """
    Multi-input disambiguation network.

    Four parallel encoder streams (numeric, soil, name embedding, date)
    are concatenated and passed through a shared decoder to a softmax head.
    Streams with no active features (n_numeric=0, n_soil=0, or
    use_name_stream=False) are skipped in both construction and forward pass.

    Parameters
    ----------
    n_numeric          : number of numeric input features
    n_soil             : number of soil probability features (typically 123)
    n_vocab            : vocabulary size for name embedding (max encoded index + 1)
    n_classes          : number of output classes
    use_name_stream    : if False, the name embedding stream is not built or used.
                         Set automatically when only one source taxon is present.
    embed_dim          : name embedding dimension
    hidden_dim         : base hidden dimension for encoders
    encoder_depth      : number of layers per stream encoder (numeric, date, name)
    soil_encoder_depth : number of layers for the soil stream encoder.
                         Default 3 -- one more than the numeric default -- to give
                         the 123-dim soil input additional compression capacity.
    decoder_depth      : number of layers in shared decoder
    dropout            : dropout rate applied throughout
    """

    def __init__(
        self,
        n_numeric: int,
        n_soil: int,
        n_lc: int,
        n_vocab: int,
        n_classes: int,
        use_name_stream: bool       = True,
        embed_dim: int              = DEFAULT_EMBED_DIM,
        hidden_dim: int             = 512,
        encoder_depth: int          = 3,
        soil_encoder_depth: int     = DEFAULT_SOIL_ENCODER_DEPTH,
        lc_encoder_depth: int       = DEFAULT_LC_ENCODER_DEPTH,
        decoder_depth: int          = 3,
        dropout: float              = DEFAULT_DROPOUT,
        ) -> None:
        super().__init__()

        enc_out = max(32, hidden_dim // 4)

        self.numeric_encoder = _make_encoder(
            n_numeric, hidden_dim, enc_out, encoder_depth, dropout, nn.ELU()
        )
        self.soil_encoder = _make_encoder(
            n_soil, hidden_dim, enc_out, soil_encoder_depth, dropout, nn.ELU()
        )
        self.lc_encoder = _make_encoder(
            n_lc, hidden_dim // 2, enc_out, lc_encoder_depth, dropout, nn.ELU()
        )
        self.date_encoder = _make_encoder(
            2, hidden_dim // 4, enc_out, encoder_depth, dropout, nn.Tanh()
        )

        self.use_name_stream = use_name_stream
        if use_name_stream:
            self.name_embedding = nn.Embedding(n_vocab, embed_dim, padding_idx=0)
            self.name_encoder   = _make_encoder(
                embed_dim, hidden_dim // 4, enc_out, encoder_depth, dropout, nn.Tanh()
            )

        concat_dim = sum([
            enc_out if n_numeric        > 0   else 0,
            enc_out if n_soil           > 0   else 0,
            enc_out if n_lc             > 0   else 0,
            enc_out,                               # date always present
            enc_out if use_name_stream        else 0,
        ])

        decoder_layers: list[nn.Module] = [
            nn.Linear(concat_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
        ]
        current = hidden_dim * 2
        for _ in range(decoder_depth - 1):
            nxt = max(n_classes * 2, current // 2)
            decoder_layers += [
                nn.Linear(current, nxt),
                nn.ELU(),
                nn.Dropout(dropout),
            ]
            current = nxt

        decoder_layers.append(nn.Linear(current, n_classes))
        self.decoder = nn.Sequential(*decoder_layers)

        self.n_numeric = n_numeric
        self.n_soil    = n_soil
        self.n_lc      = n_lc

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_soil:    torch.Tensor,
        x_lc:      torch.Tensor,
        x_date:    torch.Tensor,
        x_name:    torch.Tensor,
        ) -> torch.Tensor:
        streams = []

        if self.n_numeric > 0:
            streams.append(self.numeric_encoder(x_numeric))
        if self.n_soil > 0:
            streams.append(self.soil_encoder(x_soil))
        if self.n_lc > 0:
            streams.append(self.lc_encoder(x_lc))

        streams.append(self.date_encoder(x_date))

        if self.use_name_stream:
            name_emb = self.name_embedding(x_name)
            streams.append(self.name_encoder(name_emb))

        merged = torch.cat(streams, dim=-1)
        return self.decoder(merged)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
) -> "LoreNet":
    """
    Reconstruct a LoreNet from a checkpoint produced by train() and load
    trained weights.

    All constructor arguments are read from checkpoint["architecture"],
    which is the single authoritative source for model structure. No
    inference from column list lengths or hyperparameters is performed.

    Parameters
    ----------
    checkpoint : dict loaded from checkpoint.pt via torch.load()
    device     : target device for the reconstructed model

    Returns
    -------
    LoreNet in eval mode with weights loaded.
    """
    arch = checkpoint["architecture"]
    hp   = checkpoint["hyperparameters"]

    model = LoreNet(
        n_numeric          = arch["n_numeric"],
        n_soil             = arch["n_soil"],
        n_lc               = arch.get("n_lc", 0),
        n_vocab            = arch["n_vocab"],
        n_classes          = arch["n_classes"],
        use_name_stream    = arch["use_name_stream"],
        embed_dim          = hp["embed_dim"],
        hidden_dim         = hp["hidden_dim"],
        encoder_depth      = hp["encoder_depth"],
        soil_encoder_depth = hp.get("soil_encoder_depth", DEFAULT_SOIL_ENCODER_DEPTH),
        lc_encoder_depth   = hp.get("lc_encoder_depth", DEFAULT_LC_ENCODER_DEPTH),
        decoder_depth      = hp["decoder_depth"],
        dropout            = hp.get("dropout", DEFAULT_DROPOUT),
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_checkpoint(path: str | Path) -> dict:
    """
    Load and validate a LORE model checkpoint.

    Parameters
    ----------
    path : path to checkpoint.pt written by lore/model.py

    Returns
    -------
    Checkpoint dict. Raises ValueError if required keys are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    required = {
        "model_state_dict", "class_names", "numeric_cols",
        "soil_cols", "date_cols", "n_vocab",
        "hyperparameters", "architecture",
    }
    missing = required - set(checkpoint.keys())
    if missing:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing}. "
            f"Was this checkpoint produced by lore/model.py?"
        )
    return checkpoint


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _compute_class_weights(
    labels: np.ndarray,
    n_classes: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Normalised so weights sum to n_classes.
    """
    counts  = np.bincount(labels, minlength=n_classes).astype(float)
    counts  = np.maximum(counts, 1)  # avoid division by zero for absent classes
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


def _check_imbalance(
    labels: np.ndarray,
    class_names: list[str],
    ratio_threshold: float = IMBALANCE_WARN_RATIO,
) -> None:
    counts    = np.bincount(labels)
    max_count = counts.max()
    min_count = counts[counts > 0].min()
    ratio     = max_count / min_count
    if ratio > ratio_threshold:
        max_cls = class_names[counts.argmax()]
        min_cls = class_names[counts[counts > 0].argmin()]
        warnings.warn(
            f"Severe class imbalance detected: {max_cls} ({max_count:,} records) vs "
            f"{min_cls} ({min_count:,} records), ratio={ratio:.0f}:1. "
            f"Inverse-frequency weighting is applied but results for minority "
            f"classes should be interpreted cautiously. "
            f"Consider augmenting minority class data or reviewing geo.py outputs.",
            stacklevel=3,
        )


def _build_tensors(
    df: pd.DataFrame,
    numeric_cols: list[str],
    soil_cols: list[str],
    lc_cols: list[str],
    date_cols: list[str],
    labels: np.ndarray | None,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """
    Convert DataFrame columns to tensors on device.

    Returns (x_numeric, x_soil, x_lc, x_date, x_name[, y]).
    x_name is always built -- LoreNet.forward() ignores it when
    use_name_stream is False, so no conditional logic is needed here.
    y is omitted when labels is None (inference path).
    """
    def _to_tensor(cols: list[str], dtype=torch.float32) -> torch.Tensor:
        if not cols:
            return torch.zeros(len(df), 0, dtype=dtype, device=device)
        return torch.tensor(
            df[cols].to_numpy(dtype=np.float32), dtype=dtype, device=device
        )

    x_numeric = _to_tensor(numeric_cols)
    x_soil    = _to_tensor(soil_cols)
    x_lc      = _to_tensor(lc_cols)
    x_date    = _to_tensor(date_cols)
    x_name    = torch.tensor(
        df[NAME_FEATURE].to_numpy(dtype=np.int64),
        dtype=torch.long, device=device,
    )

    if labels is not None:
        y = torch.tensor(labels, dtype=torch.long, device=device)
        return x_numeric, x_soil, x_lc, x_date, x_name, y
    return x_numeric, x_soil, x_lc, x_date, x_name


def _run_epoch(
    model: LoreNet,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    """Run one epoch. Returns (mean_loss, accuracy)."""
    model.train(training)
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            x_num, x_soil, x_lc, x_date, x_name, y = [t.to(device) for t in batch]
            logits = model(x_num, x_soil, x_lc, x_date, x_name)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(
    features: str | Path,
    run_tag: str,
    output_dir: str | Path             = DEFAULT_OUTPUT_DIR,
    exclude_features: list[str]        = None,
    confidence_threshold: float | None = None,
    drop_nodata: bool                  = True,
    impute_nodata: bool                = False,
    val_frac: float                    = DEFAULT_VAL_FRAC,
    test_frac: float                   = DEFAULT_TEST_FRAC,
    batch_size: int                    = DEFAULT_BATCH_SIZE,
    max_epochs: int                    = DEFAULT_MAX_EPOCHS,
    patience: int                      = DEFAULT_PATIENCE,
    lr: float                          = DEFAULT_LR,
    lr_patience: int                   = None,
    dropout: float                     = DEFAULT_DROPOUT,
    hidden_dim: int                    = 512,
    encoder_depth: int                 = 3,
    soil_encoder_depth: int            = DEFAULT_SOIL_ENCODER_DEPTH,
    lc_encoder_depth: int              = DEFAULT_LC_ENCODER_DEPTH,
    decoder_depth: int                 = 3,
    embed_dim: int                     = DEFAULT_EMBED_DIM,
    device_str: str                    = "auto",
) -> dict:
    """
    Train the LORE disambiguation network.

    Parameters
    ----------
    features             : path to features.parquet
    run_tag              : run identifier (e.g. 'peromyscus_split_2026')
    output_dir           : base output directory; cache and model outputs are
                           written to <output_dir>/<run_tag>/cache/
    exclude_features     : list of feat_* column names to exclude
    confidence_threshold : if set, predictions below this softmax probability
                           are flagged as low-confidence in the output.
                           Default None (no thresholding -- confidence scores
                           always written to output).
    drop_nodata          : drop records with NaN in active features (default True)
    impute_nodata        : class-conditional mean imputation instead of dropping
    val_frac             : fraction of labeled data for validation
    test_frac            : fraction of labeled data for test
    batch_size           : mini-batch size
    max_epochs           : maximum training epochs
    patience             : early stopping patience (epochs without val improvement)
    lr                   : Adam learning rate
    lr_patience          : ReduceLROnPlateau patience (epochs without val improvement
                           before LR decay). Defaults to patience // 3.
    dropout              : dropout rate
    hidden_dim           : base hidden dimension
    encoder_depth        : layers per stream encoder (numeric, date, name)
    soil_encoder_depth   : layers for the soil stream encoder. Default 3.
    decoder_depth        : layers in shared decoder
    embed_dim            : name embedding dimension
    device_str           : 'auto', 'cpu', 'cuda', or 'mps'

    Returns
    -------
    dict with keys: model, checkpoint_path, training_log, class_names,
                    test_acc, para_rate, para_results
    """
    exclude_features = exclude_features or []
    features         = Path(features)
    run_dir          = Path(output_dir) / run_tag / "cache"
    model_dir        = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if lr_patience is None:
        lr_patience = patience // 3

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

    # ---- load features -----------------------------------------------------
    logger.info("Loading features from %s", features)
    df      = pd.read_parquet(features)
    n_total = len(df)
    logger.info("  %d total records.", n_total)

    # ---- resolve feature columns -------------------------------------------
    numeric_cols, soil_cols, lc_cols, date_cols = _resolve_feature_cols(df, exclude_features)
    all_active = numeric_cols + soil_cols + lc_cols + date_cols
    logger.info(
        "  Active features: %d numeric, %d soil, %d lc, %d date.",
        len(numeric_cols), len(soil_cols), len(lc_cols), len(date_cols),
    )

    if exclude_features:
        logger.info("  Excluded: %s", exclude_features)

    # ---- name stream: disable when only one source taxon is present ---------
    # A constant embedding carries zero discriminative signal and wastes
    # parameters. Detected here on the full dataset before splitting.
    use_name_stream = df[NAME_FEATURE].nunique() > 1
    if not use_name_stream:
        warnings.warn(
            "Only one unique value found in feat_verbatim_name_encoded. "
            "Name stream disabled -- a constant embedding contributes no "
            "gradient signal and would waste parameters.",
            stacklevel=2,
        )

    # ---- split labeled / parapatric ----------------------------------------
    labeled    = _filter_single_label(df)
    parapatric = _filter_parapatric(df)
    logger.info(
        "  %d single-label records, %d parapatric records.",
        len(labeled), len(parapatric),
    )

    # ---- nodata handling on labeled ----------------------------------------
    nodata_mask = _recompute_nodata_mask(labeled, numeric_cols, soil_cols, lc_cols, date_cols)
    n_nodata    = int(nodata_mask.sum())

    if n_nodata:
        logger.info(
            "  %d labeled records have NaN in active features (%.1f%%).",
            n_nodata, 100 * n_nodata / len(labeled),
        )
        if drop_nodata and not impute_nodata:
            labeled = labeled[~nodata_mask].copy()
            logger.info("  Dropped %d nodata records.", n_nodata)
        elif impute_nodata:
            logger.info("  Nodata records will be imputed (class-conditional means).")
        else:
            logger.warning(
                "  Neither drop_nodata nor impute_nodata set -- NaN values will "
                "propagate to tensors and produce undefined behaviour."
            )

    # ---- encode labels -----------------------------------------------------
    le          = LabelEncoder()
    y_labeled   = le.fit_transform(labeled[LABEL_COL].to_numpy())
    n_classes   = len(le.classes_)
    class_names = list(le.classes_)
    logger.info("  %d classes: %s", n_classes, class_names)

    _check_imbalance(y_labeled, class_names)

    # ---- deduplicate training records by location --------------------------
    # Records sharing identical spatial features produce identical feature
    # vectors, causing data leakage across train/test splits and
    # overrepresentation of systematic trap/survey locations in gradient
    # updates. Deduplication is applied to the labeled training pool only;
    # ambiguous records passed to predict.py are unaffected.
    #
    # Deduplication requires spatial features to be active. If lat/lon are
    # excluded the spatial identity of records is unknown and deduplication
    # is skipped with a warning.
    n_before = len(labeled)
    geo_cols = {"feat_lat", "feat_lon"}
    if geo_cols.issubset(set(numeric_cols)):
        dedup_cols = ["feat_lat", "feat_lon", "feat_sin_doy", "feat_cos_doy",
                      LABEL_COL]
        labeled = labeled.drop_duplicates(subset=dedup_cols).copy()
        n_dropped = n_before - len(labeled)
        if n_dropped:
            logger.info(
                "  Dropped %d duplicate training records (%.1f%%) sharing "
                "identical spatial and temporal features.",
                n_dropped, 100 * n_dropped / n_before,
            )
            y_labeled = le.transform(labeled[LABEL_COL].to_numpy())
    else:
        n_dropped = 0
        warnings.warn(
            "Spatial features (feat_lat, feat_lon) are excluded -- skipping "
            "duplicate record removal. Data leakage across train/test splits "
            "may be present for systematic sampling datasets.",
            stacklevel=2,
        )

    n_labeled = len(labeled)

    # ---- minimum class size guard ------------------------------------------
    combined_ho = val_frac + test_frac
    min_class_size = math.ceil(1.0 / combined_ho) + 2

    post_dedup_counts = np.bincount(y_labeled)
    too_few = {
        class_names[i]: int(c)
        for i, c in enumerate(post_dedup_counts)
        if c < min_class_size
    }
    if too_few:
        raise ValueError(
            f"After deduplication, the following classes have fewer than "
            f"{min_class_size} records (minimum to populate train/val/test splits): "
            f"{too_few}.\n"
            f"Options:\n"
            f"  1. Review geo.py outputs — range polygon may be too restrictive.\n"
            f"  2. Pass --impute-nodata to retain nodata records for minority classes.\n"
            f"  3. Remove the taxon from --dest-taxa if it is not the focus of this run."
        )

    # ---- stratified split --------------------------------------------------
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=combined_ho, random_state=42)
    train_idx, ho_idx = next(sss1.split(np.zeros(n_labeled), y_labeled))

    val_rel = val_frac / combined_ho
    sss2    = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_rel, random_state=42)
    val_idx_rel, test_idx_rel = next(
        sss2.split(np.zeros(len(ho_idx)), y_labeled[ho_idx])
    )
    val_idx  = ho_idx[val_idx_rel]
    test_idx = ho_idx[test_idx_rel]

    df_train = labeled.iloc[train_idx].copy()
    df_val   = labeled.iloc[val_idx].copy()
    df_test  = labeled.iloc[test_idx].copy()
    y_train  = y_labeled[train_idx]
    y_val    = y_labeled[val_idx]
    y_test   = y_labeled[test_idx]

    logger.info(
        "  Split: %d train / %d val / %d test.",
        len(df_train), len(df_val), len(df_test),
    )

    # ---- class-conditional imputation (training split only) ----------------
    class_means: dict[int, dict[str, float]] = {}
    if impute_nodata and n_nodata > 0:
        logger.info("  Computing class-conditional imputation means on training split...")
        class_means = _compute_class_conditional_means(
            df_train, all_active, y_train, np.unique(y_train)
        )
        df_train = _apply_imputation(df_train, all_active, y_train, class_means)
        df_val   = _apply_imputation(df_val,   all_active, y_val,   class_means)
        df_test  = _apply_imputation(df_test,  all_active, y_test,  class_means)

    # ---- class weights -----------------------------------------------------
    class_weights = _compute_class_weights(y_train, n_classes).to(device)
    logger.info("  Class weights: %s", {
        class_names[i]: round(float(class_weights[i]), 4) for i in range(n_classes)
    })

    # ---- build tensors and DataLoaders -------------------------------------
    # n_vocab is derived from the full dataset so the embedding table covers
    # all names seen at inference time, not just those in the training split.
    # n_vocab is max encoded index + 1; index 0 is reserved as padding,
    # so a single source taxon produces n_vocab=2.
    n_vocab = int(df[NAME_FEATURE].max()) + 1

    train_tensors = _build_tensors(df_train, numeric_cols, soil_cols, lc_cols, date_cols, y_train, device)
    val_tensors   = _build_tensors(df_val,   numeric_cols, soil_cols, lc_cols, date_cols, y_val,   device)
    test_tensors  = _build_tensors(df_test,  numeric_cols, soil_cols, lc_cols, date_cols, y_test,  device)

    train_ds = TensorDataset(*train_tensors)
    val_ds   = TensorDataset(*val_tensors)
    test_ds  = TensorDataset(*test_tensors)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ---- build model -------------------------------------------------------
    model = LoreNet(
        n_numeric          = len(numeric_cols),
        n_soil             = len(soil_cols),
        n_lc               = len(lc_cols),
        n_vocab            = n_vocab,
        n_classes          = n_classes,
        use_name_stream    = use_name_stream,
        embed_dim          = embed_dim,
        hidden_dim         = hidden_dim,
        encoder_depth      = encoder_depth,
        soil_encoder_depth = soil_encoder_depth,
        lc_encoder_depth   = lc_encoder_depth,
        decoder_depth      = decoder_depth,
        dropout            = dropout,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Model parameters: %d", n_params)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience,
    )

    # ---- training loop -----------------------------------------------------
    best_val_loss  = float("inf")
    best_state     = None
    epochs_no_imp  = 0
    log_rows: list[dict] = []

    print()
    print("=" * 55)
    print("  LORE -- model training")
    print("=" * 55)
    print(f"  Classes    : {n_classes}  ({', '.join(class_names)})")
    print(f"  Train      : {len(df_train):,}")
    print(f"  Val        : {len(df_val):,}")
    print(f"  Test       : {len(df_test):,}")
    print(f"  Name stream: {'enabled' if use_name_stream else 'disabled (single source taxon)'}")
    print(f"  Params     : {n_params:,}")
    print(f"  Device     : {device}")
    print(f"  Max epochs : {max_epochs}  (patience={patience}, lr_patience={lr_patience})")
    print("=" * 55)

    pbar = tqdm(range(max_epochs), desc="training", unit="epoch")
    for epoch in pbar:
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, training=False
        )
        scheduler.step(val_loss)

        log_rows.append({
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
        })

        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss":   f"{val_loss:.4f}",
            "val_acc":    f"{val_acc:.4f}",
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                logger.info("  Early stopping at epoch %d.", epoch + 1)
                break

    # ---- test evaluation ---------------------------------------------------
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_acc = _run_epoch(
        model, test_loader, criterion, None, device, training=False
    )
    logger.info("  Test -- loss=%.4f  acc=%.4f", test_loss, test_acc)

    if test_acc < 0.9:
        warnings.warn(
            f"Test accuracy {test_acc:.3f} is below 0.90. Results for minority "
            f"classes may be unreliable. Review training summary and consider "
            f"additional data for underrepresented taxa.",
            stacklevel=2,
        )

    # ---- parapatric evaluation ---------------------------------------------
    para_total    = 0
    para_resolved = 0
    para_results: list[dict] = []

    if len(parapatric) > 0:
        logger.info("  Evaluating on %d parapatric records...", len(parapatric))
        para_nodata = _recompute_nodata_mask(
            parapatric, numeric_cols, soil_cols, lc_cols, date_cols
        )
        para_eval = parapatric[~para_nodata].copy()

        if len(para_eval) > 0:
            para_tensors = _build_tensors(
                para_eval, numeric_cols, soil_cols, lc_cols, date_cols, None, device
            )
            para_ds     = TensorDataset(*para_tensors)
            para_loader = DataLoader(para_ds, batch_size=batch_size, shuffle=False)

            model.eval()
            all_probs: list[torch.Tensor] = []
            with torch.no_grad():
                for batch in para_loader:
                    x_num, x_soil, x_lc, x_date, x_name = [t.to(device) for t in batch]
                    logits = model(x_num, x_soil, x_lc, x_date, x_name)
                    all_probs.append(torch.softmax(logits, dim=1).cpu())

            probs      = torch.cat(all_probs, dim=0).numpy()
            pred_idx   = probs.argmax(axis=1)
            pred_names = [class_names[i] for i in pred_idx]
            confidence = probs.max(axis=1)

            for i, (_, row) in enumerate(para_eval.iterrows()):
                candidates  = set(row[LABEL_COL].split(LABEL_PARAPATRIC_SEP))
                is_resolved = pred_names[i] in candidates
                para_total    += 1
                para_resolved += int(is_resolved)
                para_results.append({
                    "gbifID":          row.get("gbifID", ""),
                    "suggested_names": row[LABEL_COL],
                    "ml_prediction":   pred_names[i],
                    "ml_confidence":   round(float(confidence[i]), 6),
                    "candidate_match": is_resolved,
                })

        para_rate = para_resolved / para_total if para_total > 0 else float("nan")
        logger.info(
            "  Parapatric resolution: %d / %d  (%.1f%%)",
            para_resolved, para_total, 100 * para_rate,
        )
    else:
        para_rate = float("nan")
        logger.info("  No parapatric records found for evaluation.")

    # ---- save checkpoint ---------------------------------------------------
    # architecture is the single authoritative source for LoreNet reconstruction.
    # build_model_from_checkpoint() reads from here -- nothing is inferred from
    # column list lengths or hyperparameters.
    checkpoint = {
        "model_state_dict": best_state,
        "class_names":      class_names,
        "class_encoder":    {name: int(i) for i, name in enumerate(class_names)},
        "numeric_cols":     numeric_cols,
        "soil_cols":        soil_cols,
        "lc_cols":          lc_cols,
        "date_cols":        date_cols,
        "name_feature":     NAME_FEATURE,
        "n_vocab":          n_vocab,
        "exclude_features": exclude_features,
        "class_means":      class_means,
        "confidence_threshold": confidence_threshold,
        "architecture": {
            "n_numeric":       len(numeric_cols),
            "n_soil":          len(soil_cols),
            "n_lc":            len(lc_cols),
            "n_vocab":         n_vocab,
            "n_classes":       n_classes,
            "use_name_stream": use_name_stream,
        },
        "hyperparameters": {
            "hidden_dim":          hidden_dim,
            "encoder_depth":       encoder_depth,
            "soil_encoder_depth":  soil_encoder_depth,
            "lc_encoder_depth": lc_encoder_depth,
            "decoder_depth":       decoder_depth,
            "embed_dim":           embed_dim,
            "dropout":             dropout,
            "lr":                  lr,
            "lr_patience":         lr_patience,
            "batch_size":          batch_size,
            "max_epochs":          max_epochs,
            "patience":            patience,
        },
        "training_metadata": {
            "n_train":        len(df_train),
            "n_val":          len(df_val),
            "n_test":         len(df_test),
            "n_params":       n_params,
            "epochs_trained": len(log_rows),
            "best_val_loss":  round(best_val_loss, 6),
            "test_loss":      round(test_loss, 6),
            "test_acc":       round(test_acc, 6),
            "para_total":     para_total,
            "para_resolved":  para_resolved,
            "para_rate":      round(para_rate, 4) if not np.isnan(para_rate) else None,
            "features_path":  str(features),
            "run_tag":        run_tag,
            "device":         str(device),
        },
    }

    checkpoint_path = model_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info("Checkpoint saved to %s", checkpoint_path)

    # ---- save training log -------------------------------------------------
    log_df   = pd.DataFrame(log_rows)
    log_path = model_dir / "training_log.csv"
    log_df.to_csv(log_path, index=False)
    logger.info("Training log saved to %s", log_path)

    # ---- save training summary ---------------------------------------------
    summary_lines = [
        "=" * 60,
        "  LORE -- Training Summary",
        "=" * 60,
        f"  Run tag         : {run_tag}",
        f"  Features parquet: {features}",
        f"  Device          : {device}",
        "",
        "  Data",
        f"    Total records   : {n_total:,}",
        f"    Single-label    : {len(labeled):,}",
        f"    Dedup dropped   : {n_dropped:,}  (spatial+temporal duplicates)",
        f"    Parapatric      : {len(parapatric):,}",
        f"    Train           : {len(df_train):,}",
        f"    Val             : {len(df_val):,}",
        f"    Test            : {len(df_test):,}",
        f"    Nodata dropped  : {n_nodata:,}  ({'imputed' if impute_nodata else 'dropped'})",
        "",
        "  Classes",
    ]
    counts = np.bincount(y_labeled, minlength=n_classes)
    for i, name in enumerate(class_names):
        w = float(class_weights[i])
        summary_lines.append(
            f"    {name:<45} n={counts[i]:>7,}  weight={w:.4f}"
        )

    summary_lines += [
        "",
        "  Features",
        f"    Numeric     : {numeric_cols}",
        f"    Soil        : {len(soil_cols)} classes",
        f"    Land cover  : {len(lc_cols)} classes",
        f"    Date        : {date_cols}",
        f"    Name stream : {'enabled' if use_name_stream else 'disabled (single source taxon)'}",
        f"    Excluded    : {exclude_features or 'none'}",
        "",
        "  Hyperparameters",
        f"    hidden_dim         : {hidden_dim}",
        f"    encoder_depth      : {encoder_depth}",
        f"    soil_encoder_depth : {soil_encoder_depth}",
        f"    lc_encoder_depth   : {lc_encoder_depth}",
        f"    decoder_depth      : {decoder_depth}",
        f"    embed_dim          : {embed_dim}",
        f"    dropout            : {dropout}",
        f"    lr                 : {lr}",
        f"    lr_patience        : {lr_patience}",
        f"    batch_size         : {batch_size}",
        f"    patience           : {patience}",
        "",
        "  Results",
        f"    Epochs trained  : {len(log_rows)}",
        f"    Best val loss   : {best_val_loss:.6f}",
        f"    Test loss       : {test_loss:.6f}",
        f"    Test accuracy   : {test_acc:.4f}  ({test_acc*100:.1f}%)",
    ]
    if para_total > 0:
        summary_lines += [
            f"    Parapatric n    : {para_total:,}",
            f"    Para resolved   : {para_resolved:,}  ({para_rate*100:.1f}%)",
            "    (resolved = top prediction is one of the known candidate taxa)",
        ]
    if confidence_threshold is not None:
        summary_lines.append(
            f"    Conf. threshold : {confidence_threshold}  "
            f"(predictions below this are flagged low-confidence)"
        )
    summary_lines += [
        "",
        f"  Model params    : {n_params:,}",
        f"  Checkpoint      : {checkpoint_path}",
        f"  Training log    : {log_path}",
        "=" * 60,
    ]

    summary_path = model_dir / "training_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    logger.info("Training summary saved to %s", summary_path)

    print()
    for line in summary_lines:
        print(line)

    return {
        "model":           model,
        "checkpoint_path": checkpoint_path,
        "training_log":    log_df,
        "class_names":     class_names,
        "test_acc":        test_acc,
        "para_rate":       para_rate,
        "para_results":    para_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the LORE disambiguation network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features", type=Path, required=True,
                   help="Path to features.parquet.")
    p.add_argument("--run-tag", required=True,
                   help="Run identifier. Cache and model are written to "
                        "<output-dir>/<run-tag>/cache/.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Base output directory. Cache and model outputs are written "
                        f"to <output-dir>/<run-tag>/cache/. Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--exclude-features", nargs="*", default=[],
                   help="feat_* column names to exclude from training. "
                        "Example: --exclude-features feat_lat feat_lon")
    p.add_argument("--confidence-threshold", type=float, default=None,
                   help="Softmax probability below which predictions are flagged "
                        "low-confidence. Default: None (all predictions assigned). "
                        "Confidence scores are always written to output regardless.")
    p.add_argument("--drop-nodata", action="store_true", default=True,
                   help="Drop records with NaN in active features (default).")
    p.add_argument("--impute-nodata", action="store_true", default=False,
                   help="Class-conditional mean imputation instead of dropping.")
    p.add_argument("--val-frac",           type=float, default=DEFAULT_VAL_FRAC)
    p.add_argument("--test-frac",          type=float, default=DEFAULT_TEST_FRAC)
    p.add_argument("--batch-size",         type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-epochs",         type=int,   default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--patience",           type=int,   default=DEFAULT_PATIENCE)
    p.add_argument("--lr",                 type=float, default=DEFAULT_LR)
    p.add_argument("--lr-patience",        type=int,   default=None,
                   help="ReduceLROnPlateau patience. Defaults to patience // 3.")
    p.add_argument("--dropout",            type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--encoder-depth",      type=int,   default=3)
    p.add_argument("--soil-encoder-depth", type=int,   default=DEFAULT_SOIL_ENCODER_DEPTH,
                   help="Depth of soil stream encoder. Default 3 (one deeper than "
                        "numeric/date encoders to handle the 123-dim soil input).")
    p.add_argument("--lc-encoder-depth", type=int, default=DEFAULT_LC_ENCODER_DEPTH,
                   help="Depth of land cover stream encoder. Default 3.")
    p.add_argument("--decoder-depth",      type=int,   default=3)
    p.add_argument("--embed-dim",          type=int,   default=DEFAULT_EMBED_DIM)
    p.add_argument("--device",             default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--log-level",          default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cache_dir = Path(args.output_dir) / args.run_tag / "cache"

    print("=" * 55)
    print("  LORE -- model training")
    print("=" * 55)
    print(f"  Features   : {args.features}")
    print(f"  Run tag    : {args.run_tag}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Cache dir  : {cache_dir}")
    print(f"  Excluded   : {args.exclude_features or 'none'}")
    print(f"  Device     : {args.device}")
    print("=" * 55)

    train(
        features             = args.features,
        run_tag              = args.run_tag,
        output_dir           = args.output_dir,
        exclude_features     = args.exclude_features,
        confidence_threshold = args.confidence_threshold,
        drop_nodata          = args.drop_nodata,
        impute_nodata        = args.impute_nodata,
        val_frac             = args.val_frac,
        test_frac            = args.test_frac,
        batch_size           = args.batch_size,
        max_epochs           = args.max_epochs,
        patience             = args.patience,
        lr                   = args.lr,
        lr_patience          = args.lr_patience,
        dropout              = args.dropout,
        hidden_dim           = args.hidden_dim,
        encoder_depth        = args.encoder_depth,
        soil_encoder_depth   = args.soil_encoder_depth,
        lc_encoder_depth     = args.lc_encoder_depth,
        decoder_depth        = args.decoder_depth,
        embed_dim            = args.embed_dim,
        device_str           = args.device,
    )


if __name__ == "__main__":
    main()
