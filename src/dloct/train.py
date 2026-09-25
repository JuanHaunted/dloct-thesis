"""
Train a reconstruction model.

Single GPU:   python -m dloct.train --config configs/unet_full.yaml
Multi-GPU:    torchrun --standalone --nproc_per_node=2 -m dloct.train --config configs/unet_full.yaml
Overrides:    ... --set optim.steps=2000 data.batch_size=4

Outputs go to ``<out_dir>/<name>/``: ``config.yaml``, ``log.jsonl``, ``latest.pt``, ``best.pt``
and ``previews/``. If ``latest.pt`` exists the run resumes from it, so a SLURM job that hits
its time limit can simply be resubmitted.

Fine-tuning: ``train.init_from: runs/<other>/best.pt`` starts the network (and its EMA) from
another run's EMA weights, with a fresh optimizer and schedule.

Adversarial training: an ``adv`` section (``weight``, ``mode``, ``lr``, ``start_step``, ``base``,
``n_layers``) adds a conditional PatchGAN discriminator (``models/discriminator.py``) trained with
the hinge loss; the generator loss gains ``weight · (−D(x̂))``.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from .config import load_config
from .data import EvalBScans, TrainPatches
from .evaluation import amp_dtype, evaluate
from .losses import ReconLoss
from .models.discriminator import PatchDiscriminator, d_hinge_loss, g_hinge_loss
from .models.reconstructors import build_model
from .physics import measure
from .visualize import comparison_figure


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


def fmt(d):
    return " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], action="extend", metavar="KEY=VALUE")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    sys.stdout.reconfigure(line_buffering=True)  # stream logs into SLURM output files

    rank, world, device = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(cfg["seed"] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    run_dir = Path(cfg["out_dir"]) / cfg["name"]
    if is_main:
        (run_dir / "previews").mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    dcfg, ocfg, tcfg = cfg["data"], cfg["optim"], cfg["train"]
    K = dcfg["factor"]

    model = build_model(cfg["model"]).to(device)
    divisor = math.lcm(model.divisor, K)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ocfg["ema"]))
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=ocfg["lr"], weight_decay=ocfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(ocfg["warmup_steps"], ocfg["steps"]))
    loss_fn = ReconLoss(**cfg["loss"])
    dtype = amp_dtype(tcfg.get("precision", "auto"))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    acfg = cfg.get("adv") or {}
    use_adv = bool(acfg) and acfg.get("weight", 0) > 0
    if use_adv:
        disc = PatchDiscriminator(acfg.get("mode", "amplitude"), acfg.get("base", 64), acfg.get("n_layers", 3)).to(device)
        opt_d = torch.optim.Adam(disc.parameters(), lr=acfg.get("lr", 1e-4), betas=(0.5, 0.999))
        scaler_d = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    step, best = 0, None
    ckpt_path = run_dir / "latest.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if use_adv and "disc" in ck:
            disc.load_state_dict(ck["disc"])
            opt_d.load_state_dict(ck["opt_d"])
            scaler_d.load_state_dict(ck["scaler_d"])
        step, best = ck["step"], ck.get("best")
        if is_main:
            print(f"resumed from {ckpt_path} at step {step}")
    elif tcfg.get("init_from"):
        src = torch.load(tcfg["init_from"], map_location=device, weights_only=False)
        weights = {k.removeprefix("module."): v for k, v in src["ema"].items() if k != "n_averaged"}
        model.load_state_dict(weights)
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ocfg["ema"]))
        if is_main:
            print(f"initialized from {tcfg['init_from']} (EMA weights, step {src['step']})")

    train_model = DDP(model, device_ids=[device.index]) if world > 1 else model
    train_disc = (DDP(disc, device_ids=[device.index]) if world > 1 else disc) if use_adv else None
    if tcfg.get("compile"):
        train_model = torch.compile(train_model)

    train_ds = TrainPatches(dcfg["root"], patch=tuple(dcfg["patch"]), seed=cfg["seed"] + step,
                            rank=rank, min_energy=dcfg["min_energy"], divisor=divisor,
                            sources=dcfg.get("sources"), balance=dcfg.get("balance", "sample"))
    loader = DataLoader(train_ds, batch_size=dcfg["batch_size"], num_workers=dcfg["num_workers"],
                        pin_memory=True, drop_last=True, persistent_workers=dcfg["num_workers"] > 0)
    val_ds = EvalBScans(dcfg["root"], "val", per_volume=dcfg["val_per_volume"], divisor=divisor,
                        sources=dcfg.get("sources")) if is_main else None

    if is_main:
        print(f"model={cfg['model']['type']} params={n_params / 1e6:.2f}M world={world} "
              f"train_ranges={len(train_ds.ranges)} val_bscans={len(val_ds)} K={K} amp={dtype}"
              + (f" adv={acfg}" if use_adv else ""))

    def save(path):
        state = dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                     sched=sched.state_dict(), scaler=scaler.state_dict(), step=step, best=best, cfg=cfg)
        if use_adv:
            state.update(disc=disc.state_dict(), opt_d=opt_d.state_dict(), scaler_d=scaler_d.state_dict())
        torch.save(state, path)

    def log(record):
        with open(run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def validate():
        nonlocal best
        summary, per_source, examples, _ = evaluate(ema.module, val_ds, K, device, dcfg.get("snr_db", 10.0),
                                                    amp_dtype=dtype, keep=1)
        for method, m in summary.items():
            print(f"[val {step}] {method:14s} {fmt(m)}")
        log(dict(step=step, kind="val", metrics=summary, per_source=per_source))
        name, gt, preds = examples[0]
        comparison_figure(gt, preds, run_dir / "previews" / f"step{step:07d}.png",
                          dcfg.get("snr_db", 10.0), title=f"{cfg['name']} step {step} — {name}")
        method = "model+dc" if "model+dc" in summary else "model"
        score = summary[method][tcfg["select_metric"]]
        better = best is None or (score > best if tcfg["select_mode"] == "max" else score < best)
        if better:
            best = score
            save(run_dir / "best.pt")
            print(f"[val {step}] new best {tcfg['select_metric']}={score:.4g}")
        model.train()

    model.train()
    it = iter(loader)
    t0, seen = time.time(), 0
    rng = np.random.default_rng(cfg["seed"] + 1000 * rank + step)
    while step < ocfg["steps"]:
        x = next(it).to(device, non_blocking=True)
        offset = int(rng.integers(K))
        x_meas = measure(x, K, offset)
        autocast = torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None)
        with autocast:
            x_hat = train_model(x_meas, K, offset)
        x_hat = x_hat.to(torch.complex64)
        loss, terms = loss_fn(x_hat, x, step, factor=K, offset=offset)
        adv_on = use_adv and step >= acfg.get("start_step", 0)
        if adv_on:
            # Generator side: the discriminator is frozen and used unwrapped (no DDP sync needed).
            disc.requires_grad_(False)
            with autocast:
                g_adv = g_hinge_loss(disc(x_hat, x_meas, K, offset).float())
            disc.requires_grad_(True)
            loss = loss + acfg["weight"] * g_adv
            terms["g_adv"] = g_adv.detach()

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update_parameters(model)

        if adv_on:
            # Discriminator side: ground truth is real, the (detached) reconstruction is fake.
            # One forward over both halves, as DDP requires a single forward per backward.
            with autocast:
                logits = train_disc(torch.cat([x, x_hat.detach()]), torch.cat([x_meas, x_meas]), K, offset).float()
                real, fake = logits.chunk(2)
                d_loss = d_hinge_loss(real, fake)
            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            terms.update(d_loss=d_loss.detach(), d_real=real.mean().detach(), d_fake=fake.mean().detach())
        step += 1
        seen += x.shape[0] * world

        if is_main and step % tcfg["log_every"] == 0:
            dt = time.time() - t0
            rec = dict(step=step, kind="train", loss=loss.item(), grad_norm=grad_norm.item(),
                       lr=sched.get_last_lr()[0], samples_per_s=seen / dt,
                       **{k: v.item() for k, v in terms.items()})
            print(f"[train {step}] {fmt({k: v for k, v in rec.items() if k not in ('step', 'kind')})}")
            log(rec)
            t0, seen = time.time(), 0
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at step {step}")

        if step % tcfg["val_every"] == 0 or step == ocfg["steps"]:
            if is_main:
                validate()
            if world > 1:
                dist.barrier()
        if is_main and (step % tcfg["ckpt_every"] == 0 or step == ocfg["steps"]):
            save(ckpt_path)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
