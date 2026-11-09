# -*- coding: utf-8 -*-
"""Where does a TSplineDCM training step spend its time? Read-only: builds the model from an experiment yaml,
runs a few hundred steps on split-like random rows and prints per-section times, the forward split by module,
kernel / op counts per step, and how the step time scales with the batch size. Nothing is saved.

    python tools/profile_step.py configs/exp/LPMC_tspline_tuned.yaml            # with the basis cache (default)
    python tools/profile_step.py configs/exp/LPMC_tspline_tuned.yaml --no-cache # the pre-speed-up path
    python tools/profile_step.py configs/exp/LPMC_tspline_tuned.yaml --set model_config.zhead_fused=false  # exact mode

Numbers are only comparable on an idle GPU (the paper's timings were measured under a 6-way shared GPU)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from run import load_experiment  # noqa: E402
from train_torch import _VectorizedLoader, default_dcm_loss, masked_log_softmax  # noqa: E402


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--no-cache", action="store_true", help="disable the row basis cache (model_config.basis_cache=false)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="PATH=VALUE",
                    help="override a config item, e.g. --set model_config.zhead_fused=false (repeatable)")
    ap.add_argument("--csv", default=None, help="append one row of the key numbers to this csv (for plots/fig_speedup.py); "
                    "the batch-size sweep goes to <same dir>/speedup_batch.csv (plots/fig_speedup_batch.py)")
    ap.add_argument("--batch-sizes", default="256,1024,4096,16384,65536,full",
                    help="batch sizes of the sweep (comma list; 'full' = all training rows; sizes above the training set are skipped)")
    ap.add_argument("--label", default=None, help="mode label written to the csv row (e.g. old / cache / fused)")
    args = ap.parse_args()
    row: dict = {}

    tt = load_experiment(args.experiment, sets=args.sets, out_root=os.environ.get("TSDCM_OUT", "results") + "/tmp")["trainer"]
    if args.no_cache:
        tt.mcfg.basis_cache = False
    tt.prepare(42)
    ds, model, dev = tt.ds, tt.model, tt.device
    N = len(ds)
    rng = np.random.default_rng(0)
    train_idx = rng.permutation(N)[: int(0.7 * N)]
    bs0 = int(tt.tcfg.batch_size)
    cache_mb = model.basis_cache_bytes() / 2**20 if hasattr(model, "basis_cache_bytes") else 0.0
    print(f"[cfg] {args.experiment}: rows {N}, train {len(train_idx)}, params {sum(p.numel() for p in model.parameters())}, "
          f"batch {bs0}, eval_mode {tt.tcfg.eval_mode}, alts {len(tt.dcfg.alt_names)}, "
          f"splines {sum(len(v) for v in tt.dcfg.alt_xs.values())}, basis cache {'off' if args.no_cache else f'{cache_mb:.0f} MB'}")

    def run_steps(bs, n_steps, loss_fn, sections=None):
        model.train()
        perm = torch.as_tensor(rng.permutation(train_idx))
        t_all = 0.0
        for s in range(n_steps):
            idx = perm[(s * bs) % len(perm): (s * bs) % len(perm) + bs]
            sync(); t0 = time.perf_counter()
            batch = ds.make_batch_from_indices(idx)
            sync(); t1 = time.perf_counter()
            out = model(batch)
            sync(); t2 = time.perf_counter()
            loss, _ = loss_fn(out, batch, dev)
            sync(); t3 = time.perf_counter()
            tt.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sync(); t4 = time.perf_counter()
            tt.optimizer.step()
            sync(); t5 = time.perf_counter()
            t_all += t5 - t0
            if sections is not None:
                for k, v in (("batch", t1 - t0), ("forward", t2 - t1), ("loss+metrics", t3 - t2), ("backward", t4 - t3), ("optim", t5 - t4)):
                    sections[k] += v
        return t_all / n_steps

    def loss_gpu_only(output, batch, device):
        """Same loss, metrics kept on the GPU (no sklearn f1, no .item() per step)."""
        V = output["V"]; y = batch["label"]
        logP = output["P"].clamp_min(1e-12).log() if "P" in output else masked_log_softmax(V, batch.get("av"))
        nll = -(y * logP).sum(dim=1).mean()
        loss = nll + output.get("kld", 0.0)
        if "penalty" in output:
            loss = loss + output["penalty"]
        return loss, {}

    run_steps(bs0, 5, default_dcm_loss)   # warm-up

    n = args.steps
    sec = defaultdict(float)
    per_step = run_steps(bs0, n, default_dcm_loss, sec)
    tot = sum(sec.values())
    steps_per_epoch = max(1, len(train_idx) // bs0)
    print(f"\n[1] step at batch {bs0}: {per_step*1e3:.1f} ms  ({steps_per_epoch} steps/epoch -> {per_step*steps_per_epoch:.1f} s/epoch of training)")
    for k, v in sec.items():
        print(f"     {k:14s} {v/n*1e3:7.2f} ms  {v/tot:5.1%}")
    row.update({"step_ms": per_step * 1e3, **{f"{k.replace('+', '_')}_ms": v / n * 1e3 for k, v in sec.items()},
                "steps_per_epoch": steps_per_epoch, "batch_size": bs0, "cache_mb": 0.0 if args.no_cache else cache_mb})

    times = defaultdict(float)
    handles = []

    def hook(name, mod):
        def pre(m, inp):
            sync(); m._t0 = time.perf_counter()

        def post(m, inp, out):
            sync(); times[name] += time.perf_counter() - m._t0
        handles.append(mod.register_forward_pre_hook(pre)); handles.append(mod.register_forward_hook(post))

    hook("taste_encoder", model.taste_encoder)
    for alt in tt.dcfg.alt_names:
        hook("spline utilities (all alts)", model.sp_utils[alt])
    if model.asc_learner is not None:
        hook("asc head (z)", model.asc_learner)
    sec2 = defaultdict(float)
    run_steps(bs0, n, default_dcm_loss, sec2)
    for h in handles:
        h.remove()
    fwd = sec2["forward"]
    print(f"\n[2] forward = {fwd/n*1e3:.1f} ms per step; inside it:")
    for k, v in times.items():
        print(f"     {k:38s} {v/n*1e3:7.2f} ms  {v/fwd:5.1%}")
    rest = fwd - sum(times.values())
    row.update({"fwd_taste_ms": times.get("taste_encoder", 0.0) / n * 1e3, "fwd_splines_ms": times.get("spline utilities (all alts)", 0.0) / n * 1e3,
                "fwd_asc_ms": times.get("asc head (z)", 0.0) / n * 1e3, "fwd_rest_ms": rest / n * 1e3})
    print(f"     {'rest (eps, tau, softmax, kl, penalty)':38s} {rest/n*1e3:7.2f} ms  {rest/fwd:5.1%}")

    from torch.profiler import ProfilerActivity, profile  # noqa: E402
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=acts) as prof:
        run_steps(bs0, 5, default_dcm_loss)
    ev = prof.key_averages()
    n_ops = sum(e.count for e in ev if e.key.startswith("aten::"))
    n_launch = sum(e.count for e in ev if e.key in ("cudaLaunchKernel", "cuLaunchKernel"))
    n_sync = sum(e.count for e in ev if "Synchronize" in e.key or "cudaMemcpy" in e.key)
    top = sorted(ev, key=lambda e: e.self_cpu_time_total, reverse=True)[:8]
    row.update({"aten_ops": n_ops / 5, "kernel_launches": n_launch / 5, "syncs_copies": n_sync / 5})
    print(f"\n[3] per step: about {n_ops/5:.0f} aten ops, {n_launch/5:.0f} kernel launches, {n_sync/5:.0f} syncs/copies; top self-CPU ops:")
    for e in top:
        print(f"     {e.key[:38]:38s} calls/step {e.count/5:7.0f}  self cpu {e.self_cpu_time_total/5/1e3:7.2f} ms")

    a = run_steps(bs0, n, default_dcm_loss)
    b = run_steps(bs0, n, loss_gpu_only)
    print(f"\n[4] metrics off the hot path (no sklearn f1 / .item() per step): {a*1e3:.1f} -> {b*1e3:.1f} ms per step ({a/b:.2f}x)")
    sweep = []
    sizes = sorted({len(train_idx) if b_ == "full" else int(b_) for b_ in args.batch_sizes.split(",")} | {bs0})
    for bs in [b_ for b_ in sizes if b_ <= len(train_idx)]:
        k = max(5, min(n, len(train_idx) // bs))
        run_steps(bs, 2, default_dcm_loss)                       # warm-up at this size (allocator)
        t = run_steps(bs, k, default_dcm_loss)                   # the real loss, as in training
        steps = max(1, -(-len(train_idx) // bs))                 # ceil: steps per epoch
        rows_per_s = min(bs, len(train_idx)) / t
        sweep.append({"batch_size": bs, "step_ms": t * 1e3, "steps_per_epoch": steps, "epoch_s": t * steps, "rows_per_s": rows_per_s})
        print(f"     batch {bs:6d}: {t*1e3:8.1f} ms/step -> {t*steps:6.2f} s per epoch of training ({rows_per_s/1e3:7.1f}k rows/s)")

    val_idx = rng.permutation(N)[: int(0.15 * N)]
    vl = _VectorizedLoader(ds, val_idx, bs0, shuffle=False)
    sync(); t0 = time.perf_counter(); tt.evaluate(vl); sync(); t_val = time.perf_counter() - t0
    vl2 = _VectorizedLoader(ds, val_idx, len(val_idx), shuffle=False)
    sync(); t0 = time.perf_counter(); tt.evaluate(vl2); sync(); t_val2 = time.perf_counter() - t0
    print(f"\n[5] validation pass ({len(val_idx)} rows, mode {tt.tcfg.eval_mode}): {t_val:.2f} s at batch {bs0}, {t_val2:.2f} s as one batch")
    gu = {"enabled": True, "every": 1, "warmup": 1, "epoch_cap": 100, "sample_per_x": len(ds)}
    sync(); t0 = time.perf_counter(); tt._maybe_update_grid(1, train_idx, gu); sync()
    t_gu = time.perf_counter() - t0
    row.update({"val_s_batched": t_val, "val_s_one_batch": t_val2, "grid_update_s": t_gu})
    if args.csv:
        import pandas as pd
        row = {"experiment": Path(args.experiment).stem, "label": args.label or ("old" if args.no_cache else "cache"),
               "sets": " ".join(args.sets), "rows": N, **row}
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"appended to {out}")
        outb = out.with_name("speedup_batch.csv")
        pd.DataFrame([{"experiment": row["experiment"], "label": row["label"], "rows": N, "train_rows": len(train_idx), **r}
                      for r in sweep]).to_csv(outb, mode="a", header=not outb.exists(), index=False)
        print(f"appended {len(sweep)} rows to {outb}")
    sync(); t0 = time.perf_counter(); run_steps(bs0, 1, loss_gpu_only); sync()
    print(f"     grid update (all splines): {t_gu:.2f} s; first step after it (cache rebuild included): {(time.perf_counter()-t0)*1e3:.0f} ms")


if __name__ == "__main__":
    main()
