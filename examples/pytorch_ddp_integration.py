"""Failure-aware checkpointing in a PyTorch DDP training loop.

Replaces the usual fixed schedule

    if step % ckpt_every == 0: save(...)

with OnlineCheckpointScheduler, which times its own saves to learn the
checkpoint cost C, counts restarts persisted inside the checkpoint to learn the
run's MTBF, and checkpoints at Daly's interval (or the Weibull-adaptive one).

Run on one machine (CPU, gloo backend), with demo-scale settings so checkpoints
happen every few seconds instead of every ~20 minutes:
    torchrun --nproc_per_node=2 examples/pytorch_ddp_integration.py --steps 100000 \
        --prior-mtbf-h 0.005 --init-cost-s 0.05 --min-interval-s 1
Kill it part-way (Ctrl-C) and rerun the same command: it resumes from the last
checkpoint, counts the interruption, and shortens its interval.
(On Windows builds without libuv, launch each rank with RANK/WORLD_SIZE/
MASTER_ADDR/MASTER_PORT set and USE_LIBUV=0 instead of torchrun.)
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from llm_reliability.checkpointing import OnlineCheckpointScheduler


def save_ckpt(path, model, opt, step, sched):
    tmp = path + ".tmp"
    torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "step": step,
                "ckpt_sched": sched.state_dict()}, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-write never leaves a torn checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ckpt", default="ddp_example_ckpt.pt")
    ap.add_argument("--prior-mtbf-h", type=float, default=10.0,
                    help="fleet prior for system MTBF at this scale (hours)")
    ap.add_argument("--weibull-shape", type=float, default=None,
                    help="set < 1 to checkpoint densely right after restarts")
    ap.add_argument("--init-cost-s", type=float, default=30.0,
                    help="assumed save cost before the first save is timed")
    ap.add_argument("--min-interval-s", type=float, default=5.0,
                    help="floor on the interval (60 s+ in production; small here for the demo)")
    args = ap.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(0)

    model = DDP(nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 1)))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = OnlineCheckpointScheduler(prior_mtbf_h=args.prior_mtbf_h, weibull_shape=args.weibull_shape,
                                      init_cost_s=args.init_cost_s, min_interval_s=args.min_interval_s)

    start = 0
    if os.path.exists(args.ckpt):                      # resume after an interruption
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.module.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"] + 1
        sched.load_state_dict(ck["ckpt_sched"])
        sched.on_restart()
        if rank == 0:
            print(f"resumed at step {start}; restarts so far = {sched.n_failures}, "
                  f"MTBF estimate = {sched.mtbf_h * 60:.1f} min")

    for step in range(start, args.steps):
        x = torch.randn(32, 64)
        loss = (model(x) - x.sum(1, keepdim=True)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        time.sleep(0.01)                                # stand-in for real step time

        # every rank must take the same decision, so broadcast rank 0's choice
        flag = torch.tensor([int(sched.should_checkpoint())])
        dist.broadcast(flag, src=0)
        if flag.item():
            with sched.timed_save():
                if rank == 0:
                    save_ckpt(args.ckpt, model, opt, step, sched)
                dist.barrier()
            if rank == 0:
                print(f"step {step:5d} loss {loss.item():.3f}  checkpoint "
                      f"(next in {sched.interval_h()*3600:.0f} s, C = {sched.cost_h*3600:.2f} s)")

    sched.mark_clean_exit()
    if rank == 0:
        save_ckpt(args.ckpt, model, opt, args.steps - 1, sched)
        print("finished; clean exit recorded")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
