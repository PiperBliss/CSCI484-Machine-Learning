# Experiments orchestrator

Use `run_experiments.py` as the single entry point to convert datasets, run training per dataset, merge datasets, and produce comparison plots.

Basic usage:

```bash
python3 run_experiments.py \
  --datasets carpet metal_nut toothbrush \
  --work_dir experiments \
  --epochs 30 \
  --batch_size 32 \
  --num_workers 4 \
  --image_size 224
```

Outputs (under `--work_dir`):
- `converted_<name>/` - converted ImageFolder dataset (train/val/test with accept/reject)
- `results_<name>/` - per-run outputs from `carpet_train.py` (metrics.csv, curves, confusion matrices)
- `converted_merged/` - merged dataset of all converted datasets
- `results_merged/` - outputs for merged training
- `summary_plots/` - labeled comparison plots and `summary_metrics.csv`
- `logs/` - per-run stdout logs written next to results folders

See `requirements.txt` for minimal Python packages.
