# Running on EAFIT Apolo

Facts below come from the [Apolo user docs](https://github.com/eafit-apolo/apolo-users/tree/master/docs).
Those docs are partly outdated and silent on several points. Items marked **ask staff** need
confirmation from apolo@eafit.edu.co.

## What we know

| | |
|---|---|
| Access | VPN required (GlobalProtect portal `leto.omega.eafit.edu.co`, or `vpnc` on Linux), then `ssh ugr-jcospinav@apolo-3.eafit.edu.co` |
| GPUs | partition `accel-2`: one node with 2× Tesla V100 (compute capability 7.0: no bf16, so training uses fp16 automatically) |
| Old GPUs | partition `accel`, Tesla K80. Too old for this PyTorch build, do not use |
| Time limit | `--time` is mandatory (format `D-HH:MM:SS`). Max walltime: **ask staff** |
| Per-user cap | 96 CPUs and 192 GB memory (QOS) |
| Account string | none mentioned in the docs |
| Storage | home is `/home/<user>`. Scratch location, quotas and where a 30 GB dataset should live: **ask staff** |
| Internet on nodes | not documented. `setup_cluster.sh` detects it and falls back to an offline wheel bundle |
| Driver | not documented. Our PyTorch build (CUDA 12.6) needs driver ≥ 525. Check with `nvidia-smi` on a GPU node |

Questions to email staff:
1. What is the max walltime on `accel-2`?
2. Where should a ~30 GB dataset live (home quota or a scratch path)?
3. Do the login and compute nodes have internet access (pip/PyPI)?
4. What is the NVIDIA driver version on `accel-2`?

## 0. Probe the cluster

`scripts/apolo_probe.sh` answers most of the open questions above. It is read-only and writes
`~/apolo_report.txt`:

```bash
scp scripts/apolo_probe.sh ugr-jcospinav@apolo-3.eafit.edu.co:~/
ssh ugr-jcospinav@apolo-3.eafit.edu.co 'bash ~/apolo_probe.sh'
scp ugr-jcospinav@apolo-3.eafit.edu.co:~/apolo_report.txt .
```

## 1. Upload code and data (from your machine, VPN on)

```bash
# code (no venv, data or runs)
rsync -avP --exclude .venv --exclude data --exclude runs --exclude wheelhouse \
    ./ ugr-jcospinav@apolo-3.eafit.edu.co:~/dloct-thesis/

# raw real tomograms only; --partial lets an interrupted transfer resume
rsync -avP --partial /path/to/dataset/phase/ ugr-jcospinav@apolo-3.eafit.edu.co:~/dloct/raw/phase/
```

If the login node has internet, `git clone` of the repo works instead of the first rsync.
The data never needs internet: it is uploaded once and training reads it from disk.

## 2. Environment (login node, once)

```bash
cd ~/dloct-thesis
bash scripts/setup_cluster.sh
```

- **Online:** installs `uv` and the locked environment (`.venv`).
- **Offline:** the script stops and asks for a wheel bundle. Build it on your machine and
  upload it, then rerun the script. It then uses the `python/3.12_miniconda-24.7.1` module:

  ```bash
  bash scripts/build_wheelhouse.sh                                   # local, ~3 GB
  rsync -avP wheelhouse ugr-jcospinav@apolo-3.eafit.edu.co:~/dloct-thesis/
  ```

Check the GPU and driver once:

```bash
srun -p accel-2 --gres=gpu:1 -t 0-00:10:00 --pty bash -c \
    'nvidia-smi; source scripts/env.sh; python -c "import torch; print(torch.cuda.get_device_name())"'
```

## 3. Prepare data (once, CPU job)

```bash
export DLOCT_DATA=$HOME/dloct/prepared      # or the scratch path staff give you
sbatch scripts/prepare.slurm $HOME/dloct/raw $DLOCT_DATA
```

The raw directory must contain one subfolder per source (`phase/`, and optionally
`synthetic/`). The job log prints per-volume diagnostics: lateral coherence before and after
bulk-phase correction. Keep that log for the thesis.

## 4. Train and evaluate

```bash
sbatch scripts/train.slurm configs/unet_full.yaml
sbatch scripts/train.slurm configs/cascade_full.yaml
sbatch scripts/train.slurm configs/unet_magnitude.yaml     # ablation
squeue -u $USER                                            # status
tail -f slurm-dloct-<jobid>.out                            # live log
sbatch scripts/eval.slurm runs/unet_full                   # after training
```

- If a job hits its time limit, resubmit the same command. It resumes from `runs/<name>/latest.pt`.
- The node has 2 GPUs. Two jobs of `--gres=gpu:2` run one after the other; to run two experiments at once, submit each with `sbatch --gres=gpu:1 ...`.
- Copy results back with `rsync -avP ugr-jcospinav@apolo-3.eafit.edu.co:~/dloct-thesis/runs/ ./runs/`.
