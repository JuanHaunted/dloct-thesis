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
