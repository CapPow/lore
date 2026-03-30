"""
sweep.py -- hyperparameter sweep script for LORE model training.

Runs a fixed grid of configurations sequentially against the Peromyscus
features parquet (geo-features excluded). After each run, reads
training_metadata from the checkpoint and accumulates a results table.

Each config writes to its own run-tag under runs/ so checkpoints and
training logs are preserved for inspection.

DEFAULT_FEATURES, and BASE_RUN_TAG need altered before swapping test
cases. Also note, this script presumes the pre-reqs exist. If starting
from scratch, the easiest option is probably to just run_pipeline on
your target test set first to ensure the features are in place.

Usage:
    python sweep.py
    python sweep.py --output-dir runs --features runs/peromyscus_split_2026/features.parquet
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
# Each dict maps directly to lore/model.py CLI flags.
# All configs exclude lat/lon to isolate non-geographic discriminability.

SWEEP_CONFIGS = [
    {
        "label":              "A_baseline",
        "hidden_dim":         256,
        "encoder_depth":      2,
        "soil_encoder_depth": 2,   # old default -- true baseline
        "dropout":            0.1,
    },
    {
        "label":              "B_soil_fix",
        "hidden_dim":         256,
        "encoder_depth":      2,
        "soil_encoder_depth": 3,   # soil encoder depth fix only
        "dropout":            0.1,
    },
    {
        "label":              "C_bigger",
        "hidden_dim":         512,
        "encoder_depth":      2,
        "soil_encoder_depth": 3,
        "dropout":            0.1,
    },
    {
        "label":              "D_deeper",
        "hidden_dim":         512,
        "encoder_depth":      3,
        "soil_encoder_depth": 4,
        "dropout":            0.1,
    },
    {
        "label":              "E_dropout",
        "hidden_dim":         512,
        "encoder_depth":      3,
        "soil_encoder_depth": 4,
        "dropout":            0.25,
    },
]

EXCLUDE_FEATURES = ["feat_lat", "feat_lon"]

#DEFAULT_FEATURES   = "runs/peromyscus_split_2026/features.parquet"
DEFAULT_FEATURES   = "runs/chaetodipus_nelsoni_split_2026/features.parquet"
DEFAULT_OUTPUT_DIR = "runs"
#BASE_RUN_TAG       = "peromyscus_sweep"
BASE_RUN_TAG       = "chaetodipus_sweep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_cmd(cfg: dict, features: str, output_dir: str) -> list[str]:
    run_tag = f"{BASE_RUN_TAG}_{cfg['label']}"
    cmd = [
        sys.executable, "-m", "lore.model",
        "--features",           features,
        "--run-tag",            run_tag,
        "--output-dir",         output_dir,
        "--exclude-features",   *EXCLUDE_FEATURES,
        "--hidden-dim",         str(cfg["hidden_dim"]),
        "--encoder-depth",      str(cfg["encoder_depth"]),
        "--soil-encoder-depth", str(cfg["soil_encoder_depth"]),
        "--dropout",            str(cfg["dropout"]),
    ]
    return cmd


def read_results(cfg: dict, output_dir: str) -> dict:
    run_tag   = f"{BASE_RUN_TAG}_{cfg['label']}"
    ckpt_path = Path(output_dir) / run_tag / "cache" / "model" / "checkpoint.pt"
    if not ckpt_path.exists():
        return {"error": f"checkpoint not found: {ckpt_path}"}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("training_metadata", {})
    hp   = ckpt.get("hyperparameters", {})
    return {
        "label":              cfg["label"],
        "hidden_dim":         hp.get("hidden_dim"),
        "encoder_depth":      hp.get("encoder_depth"),
        "soil_encoder_depth": hp.get("soil_encoder_depth"),
        "dropout":            hp.get("dropout"),
        "epochs_trained":     meta.get("epochs_trained"),
        "best_val_loss":      meta.get("best_val_loss"),
        "test_acc":           meta.get("test_acc"),
        "para_total":         meta.get("para_total"),
        "para_resolved":      meta.get("para_resolved"),
        "para_rate":          meta.get("para_rate"),
        "n_params":           meta.get("n_params"),
    }


def print_table(rows: list[dict]) -> None:
    if not rows:
        return

    col_order = [
        "label", "hidden_dim", "encoder_depth", "soil_encoder_depth",
        "dropout", "epochs_trained", "best_val_loss", "test_acc",
        "para_rate", "para_resolved", "para_total", "n_params",
    ]

    # Build display rows
    display = []
    for r in rows:
        display.append({k: str(r.get(k, "")) for k in col_order})

    # Column widths
    widths = {k: max(len(k), max(len(d[k]) for d in display)) for k in col_order}

    sep  = "  ".join("-" * widths[k] for k in col_order)
    hdr  = "  ".join(k.ljust(widths[k]) for k in col_order)

    print()
    print("=" * len(sep))
    print("  LORE -- sweep results")
    print("=" * len(sep))
    print(hdr)
    print(sep)
    for d in display:
        print("  ".join(d[k].ljust(widths[k]) for k in col_order))
    print("=" * len(sep))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="LORE hyperparameter sweep.")
    p.add_argument("--features",   default=DEFAULT_FEATURES,
                   help=f"Path to features.parquet. Default: {DEFAULT_FEATURES}")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Base output directory. Default: {DEFAULT_OUTPUT_DIR}")
    args = p.parse_args()

    features   = args.features
    output_dir = args.output_dir

    if not Path(features).exists():
        print(f"ERROR: features parquet not found: {features}", flush=True)
        print("Run the full pipeline through step 4 first.", flush=True)
        sys.exit(1)

    print(f"Sweep: {len(SWEEP_CONFIGS)} configs", flush=True)
    print(f"Features: {features}", flush=True)
    print(f"Excluded: {EXCLUDE_FEATURES}", flush=True)
    print(flush=True)

    results = []
    for i, cfg in enumerate(SWEEP_CONFIGS):
        print(f"[{i+1}/{len(SWEEP_CONFIGS)}] Running config: {cfg['label']}", flush=True)
        cmd = build_cmd(cfg, features, output_dir)
        print("  " + " ".join(cmd), flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0
        print(f"  Finished in {elapsed:.0f}s  (exit code {proc.returncode})", flush=True)

        row = read_results(cfg, output_dir)
        if "error" in row:
            print(f"  WARNING: {row['error']}", flush=True)
        else:
            results.append(row)
            print(
                f"  test_acc={row['test_acc']}  "
                f"para_rate={row['para_rate']}  "
                f"val_loss={row['best_val_loss']}  "
                f"epochs={row['epochs_trained']}",
                flush=True,
            )
        print(flush=True)

    print_table(results)


if __name__ == "__main__":
    main()
