# Data

Contents of this directory are git-ignored (except this file). The full dataset (~30 GB)
lives on Google Drive and on the cluster; keep only a small sample here for local testing.

Expected layout:

```
data/
  train/
    synthetic/   *.npy   simulated, Nyquist-sampled tomograms
    phase/       *.npy   real acquisitions
```

Each `.npy` is a complex tomogram stored either as a complex dtype or as a real array with a
trailing axis of size 2 holding (Re, Im).

## Getting the data from Google Drive

On the cluster (or here), `rclone` is the most reliable option for tens of GB:

```bash
curl https://rclone.org/install.sh | sudo bash    # or download the static binary into ~/bin without sudo
rclone config                                      # new remote "gdrive", type "drive"; on a headless
                                                   # login node answer "n" to auto config and follow the link
rclone copy gdrive:path/to/dataset /scratch/$USER/dloct/raw --progress --transfers 8
```

For a single shared folder link, `uvx gdown --folder <link>` also works, but it is
rate-limited on large folders.

Then convert once (writes B-scan-major complex64 plus `meta.json` with scales and splits):

```bash
uv run python -m dloct.prepare_data --src /scratch/$USER/dloct/raw --out /scratch/$USER/dloct/prepared
```

`--src` must contain one subdirectory per data source (e.g. `synthetic/`, `phase/`);
metrics are reported per source.
