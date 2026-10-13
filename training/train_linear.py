"""Train the linear formulation's network: models/linear/linear.pt.

    python training/train_linear.py [--hidden 8] [--seeds 4]

Data: the reference element with node A displaced by d along x (d in -1..1) for nu in 0..0.45, a 41 x 41 grid;
targets are the twelve reaction forces per unit E. About 15 s on a laptop CPU.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # the library lives one folder up

from neural_elements.simulation import network_path          # noqa: E402
from neural_elements.training import TrainConfig, train_linear   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    print(f"training the linear formulation's network (2-{a.hidden}-12) on single-node load cases")
    net, rep = train_linear(TrainConfig(hidden=a.hidden, seeds=a.seeds))
    out = network_path("linear", "linear")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    net.save(out, **rep)
    json.dump(rep, open(out.replace(".pt", ".json"), "w"), indent=1)
    e = rep["error"]
    print(f"{rep['weights']} weights, {rep['seconds']:.0f} s; reaction-force error on new load cases: "
          f"median {e['median']:.3f}%, 95th percentile {e['p95']:.3f}%\nsaved {out}")
