# Running LORE on ASU Sol HPC

This guide is written for ASU Sol. Sections 3 onward apply to any SLURM-based
cluster; a summary of site-specific items to adapt is at the end.

**First-run time budget:** plan for approximately 2 hours of data downloading
and 2 hours of raster preprocessing before the pipeline steps begin. A first
run requires up to 10 hours of wall time. Subsequent runs skip downloading
and preprocessing and are significantly faster.

---

## 1. Access the terminal

Go to https://sol.asu.edu/ and navigate to **System > Sol Shell Access**.

Sol requires either the ASU VPN or a campus network connection. Some on-campus
connections may still require VPN; if the portal is unreachable, connect to
the ASU VPN and try again.

---

## 2. Request an interactive GPU session

From the login node, run:

```bash
salloc --partition=public --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=10:00:00
```

Once the allocation is granted your prompt will change to reflect the compute
node name. All subsequent commands run on that node.

Parameter notes:

- `--partition=public` — Sol public GPU partition. Other clusters: check
  available partitions with `sinfo`.
- `--cpus-per-task=8` — matches `--workers 8` in the pipeline examples.
- `--mem=64G` — required for soil raster preprocessing (~44 GB peak). Do not
  reduce below 48G.
- `--time=10:00:00` — 10 hours; required for a first full run including data
  downloads and raster preprocessing. Subsequent runs can use a shorter
  allocation. Raster processing time depends on spatial extent of example taxa.

---

## 3. Load required modules

```bash
module purge
module load mamba/latest
module load cuda-12.4.1-gcc-12.1.0
```

Module names are site-specific. Other clusters: use `module avail` to find
equivalent Conda/Mamba and CUDA 12.x modules.

---

## 4. Navigate to scratch storage

LORE's downloaded rasters and run intermediates are large. Work from scratch
storage to avoid home directory quota issues.

```bash
cd /scratch/<your_netid>
```

Replace `<your_netid>` with your HPC username (e.g. `jsmith123`).

---

## 5. Clone the repository

```bash
git clone https://github.com/CapPow/lore.git
cd lore
```

---

## 6. Create and activate the environment

The environment must be created in scratch rather than the default home
directory location, which has insufficient quota for PyTorch and dependencies.

```bash
mamba create -p /scratch/<your_netid>/envs/LORE python=3.11 -y
source activate /scratch/<your_netid>/envs/LORE
pip install -r requirements.txt
```

To reactivate the environment in a future session, load modules first then
activate by path:

```bash
module purge
module load mamba/latest
module load cuda-12.4.1-gcc-12.1.0
source activate /scratch/<your_netid>/envs/LORE
```

Verify the GPU is visible before proceeding:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

This should print `True`. If it prints `False`, confirm the CUDA module loaded
without errors in step 3.

---

## 7. Run the pipeline

Global raster data (WorldClim, soil) download on the first run and are cached
in `lore/data/rasters/`. This is a one-time cost; subsequent runs skip it.

If you are reproducing the examples, the GBIF DOIs are ready to use in
`docs/gbif_datasets.txt`. For your own taxon, see the Quick Start section of
the README for instructions on creating a GBIF occurrence download.

Use `nohup` to ensure the run survives a terminal disconnect. Progress is
written to a log file you can monitor from any session:

```bash
nohup python run_pipeline.py \
    --run-tag peromyscus_split_2026 \
    --gbif-doi 10.15468/dl.3cv9hy \
    --source-taxa "Peromyscus maniculatus" \
    --dest-taxa "Peromyscus maniculatus" "Peromyscus sonoriensis" \
                "Peromyscus gambelii" "Peromyscus keeni" \
                "Peromyscus labecula" "Peromyscus arcticus" \
    --mdd-group Rodentia \
    --device cuda \
    --workers 8 \
    > runs/peromyscus_split_2026/pipeline.log 2>&1 &
```

Monitor progress:

```bash
tail -f runs/peromyscus_split_2026/pipeline.log
```

---

## 8. Locate the results

When the pipeline completes, outputs are in `runs/peromyscus_split_2026/`:

- `disambiguated.csv` — final taxon assignments for all occurrence records
- `figures/map.png` — spatial figure of resolved occurrences over range maps
- `analysis/analysis_report.txt` — feature discriminability report

---

## Notes for other SLURM clusters

| Sol-specific item | What to check at your site |
|---|---|
| `sol.asu.edu` browser terminal | Your HPC portal and shell access method |
| `--partition=public` | `sinfo` or your site documentation |
| `module load mamba/latest` | `module avail` or your site's Conda docs |
| `module load cuda-12.4.1-gcc-12.1.0` | `module avail cuda` |
| `/scratch/<your_netid>` | Your site's scratch path convention |
