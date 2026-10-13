"""Training of both networks: data on the reference element, a relative-accuracy loss, full-batch L-BFGS.

Both networks are trained the same way:
    - the targets are reaction forces of the reference element (never forces of a simulation or a test scene);
    - each squared error is weighted by (|f| / f_max + 0.1)^-2, so that small forces are learned to the same relative
      accuracy as large ones (small strains matter as much as large ones);
    - the network is compared exactly as the elements use it: relative to its own answer at rest (the offset);
    - several random starts are trained side by side; the best is kept;
    - L-BFGS with a strong-Wolfe line search, stopped once the loss improves by less than 0.1% over 500 iterations.
"""
import time
from dataclasses import asdict, dataclass

import torch

from . import linear, nonlinear


@dataclass
class TrainConfig:
    hidden: int = 8           # tanh units
    seeds: int = 4            # random starts trained side by side
    iters: int = 6000         # upper bound on L-BFGS iterations
    window: int = 500         # stop when the loss improves by less than `tol` over this many iterations
    tol: float = 1e-3
    floor: float = 0.1        # relative-accuracy weight floor
    nu: tuple = (0.0, 0.45)   # trained range of Poisson's ratio
    samples: int = 20000      # nonlinear: number of reference deformations
    stretch: tuple = (0.65, 1.45)   # nonlinear: range of the principal stretches
    seed: int = 0


def relative_weights(f, floor):
    """(|f| / f_max + floor)^-2, normalized to mean 1, per row of f (n, 12)."""
    norm = f.norm(dim=1, keepdim=True)
    w = 1.0 / (norm / norm.max() + floor) ** 2
    return w / w.mean()


def lbfgs(net, per_seed_loss, cfg, log):
    """Minimize the sum of the per-seed losses; return the index of the best seed and the loss history."""
    opt = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=cfg.window, history_size=50, tolerance_grad=1e-14,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = per_seed_loss().sum()
        loss.backward()
        return loss

    t0, last, it, history = time.time(), None, 0, []
    while it < cfg.iters:
        opt.step(closure)
        it += cfg.window
        with torch.no_grad():
            L = float(per_seed_loss().min())
        history.append((it, L))
        log(f"  iteration {it:5d}   loss {L:.3e}   ({time.time() - t0:.0f} s)")
        if last is not None and (last - L) / last < cfg.tol:
            break
        last = L
    with torch.no_grad():
        per = per_seed_loss()
    return int(per.argmin()), dict(iterations=it, seconds=time.time() - t0, loss=float(per.min()), history=history)


def train_linear(cfg=TrainConfig(), log=print):
    """The linear formulation's network on the (nu, d) grid of single-node load cases."""
    x, f = linear.training_data(cfg.nu)
    net = linear.LinearNetwork(cfg.hidden, cfg.seeds, cfg.seed)
    with torch.no_grad():
        net.xlo.copy_(x.min(0).values), net.xhi.copy_(x.max(0).values)
        net.ylo.copy_(f.min(0).values), net.yhi.copy_(f.max(0).values)
    X = net.scale_inputs(x)
    X0 = net.scale_inputs(torch.stack([x[:, 0], torch.zeros_like(x[:, 0])], 1))     # the same nu at d = 0
    Y = 2 * (f - net.ylo) / (net.yhi - net.ylo) - 1
    Y0 = 2 * (0 - net.ylo) / (net.yhi - net.ylo) - 1
    w = 1.0 / (f.abs() / f.abs().max(0).values + cfg.floor) ** 2                    # per output component
    w = w / w.mean(0)
    per_seed = lambda: (w * ((net.normalized(X) - net.normalized(X0)) - (Y - Y0)) ** 2).mean((1, 2))
    best, rep = lbfgs(net, per_seed, cfg, log)
    net = net.select(best)
    rep.update(config=asdict(cfg), error=linear_error(net, cfg.nu), weights=count(net))
    net.meta = rep
    return net, rep


def train_nonlinear(model, cfg=TrainConfig(hidden=16), log=print):
    """The nonlinear formulation's network for a material model on random reference deformations."""
    z, F, f = nonlinear.training_data(model, cfg.samples, cfg.nu, cfg.stretch, cfg.seed)
    net = nonlinear.NonlinearNetwork(model, cfg.hidden, cfg.seeds, cfg.seed)
    x = torch.cat([z, nonlinear.invariants(F)[0]], 1)
    with torch.no_grad():
        net.xlo.copy_(x.min(0).values), net.xhi.copy_(x.max(0).values)
    w, scale = relative_weights(f, cfg.floor), f.abs().max()
    per_seed = lambda: (w * ((nonlinear.reference_forces(net.stress(z, F)) - f) / scale) ** 2).mean((1, 2))
    best, rep = lbfgs(net, per_seed, cfg, log)
    net = net.select(best)
    rep.update(config=asdict(cfg), error=nonlinear_error(net, cfg), weights=count(net))
    net.meta = rep
    return net, rep


def count(net):
    return sum(p.numel() for p in net.parameters())


@torch.no_grad()
def linear_error(net, nu=(0.0, 0.45)):
    """Median and 95th percentile relative error (%) of the reaction forces on a finer grid than the training grid."""
    x, f = linear.training_data(nu, n_nu=37, n_d=37)
    x, f = x[x[:, 1] != 0], f[x[:, 1] != 0]
    p = net(x[:, 0], x[:, 1]) - net(x[:, 0], torch.zeros_like(x[:, 1]))
    e = 100 * (p - f).norm(dim=1) / f.norm(dim=1)
    return dict(median=float(e.median()), p95=float(torch.quantile(e, 0.95)))


@torch.no_grad()
def nonlinear_error(net, cfg):
    """Median and 95th percentile relative error (%) of the reaction forces on new reference deformations."""
    z, F, f = nonlinear.training_data(net.model, 4000, cfg.nu, cfg.stretch, seed=99)
    e = 100 * (nonlinear.reference_forces(net.stress(z, F)[0]) - f).norm(dim=1) / f.norm(dim=1)
    return dict(median=float(e.median()), p95=float(torch.quantile(e, 0.95)))
