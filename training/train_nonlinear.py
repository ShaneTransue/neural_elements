"""Train the nonlinear formulation's network for a material model: models/nonlinear/<material>.pt.

    python training/train_nonlinear.py --material neohookean     linear | neohookean | stvk | blend
    python training/train_nonlinear.py --all                      all four, in parallel

Data: 20,000 reference deformations (principal stretches 0.65 - 1.45 in random directions, nu in 0 - 0.45, and
kappa in 0 - 1 for the blend); targets are the reaction forces per unit E. A few minutes each on a laptop CPU.
The blend covers a whole family of materials with one network, so it gets more hidden units (32 instead of 16).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # the library lives one folder up

HIDDEN = {"linear": 16, "neohookean": 16, "stvk": 16, "blend": 32}


def train(material, hidden=None, seeds=4, threads=None, quiet=False):
    import torch
    if threads:
        torch.set_num_threads(threads)
    from neural_elements.simulation import network_path
    from neural_elements.training import TrainConfig, train_nonlinear
    net, rep = train_nonlinear(material, TrainConfig(hidden=hidden or HIDDEN[material], seeds=seeds),
                               log=(lambda *_: None) if quiet else print)
    out = network_path("nonlinear", material)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    net.save(out, **rep)
    json.dump(rep, open(out.replace(".pt", ".json"), "w"), indent=1)
    return material, rep, out


def report(material, rep, out):
    e = rep["error"]
    print(f"{material:10s} {rep['weights']} weights, {rep['seconds']:.0f} s; reaction-force error on new deformations: "
          f"median {e['median']:.2f}%, 95th percentile {e['p95']:.2f}%   -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="neohookean", choices=list(HIDDEN))
    ap.add_argument("--all", action="store_true", help="train every material model in parallel")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    if a.all:
        print("training the nonlinear formulation's networks: " + ", ".join(HIDDEN) + " (in parallel)")
        with ProcessPoolExecutor(len(HIDDEN)) as pool:
            jobs = [pool.submit(train, m, None, a.seeds, max(1, (os.cpu_count() or 4) // len(HIDDEN)), True)
                    for m in HIDDEN]
            for j in jobs:
                report(*j.result())
    else:
        print(f"training the nonlinear formulation's network for {a.material}")
        report(*train(a.material, a.hidden, a.seeds))
