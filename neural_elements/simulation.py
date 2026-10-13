"""A simulation: one scene, one force model, one integrator. Steps a frame at a time and records measurements.

    sim = Simulation("stretch", model="ne_nonlinear", material=Material("neohookean"))
    for _ in range(sim.scene.loading_frames):
        sim.step()
    print(max(abs(f) for f in sim.history["force"]))

Force models:
    ne_nonlinear   Neural Elements, nonlinear formulation (any material model)
    ne_linear      Neural Elements, linear formulation (linear material only)
    fem            finite elements of the material model (linear corotational, Neo-Hookean, St. Venant-Kirchhoff,
                   or blend)
    mass_spring    springs on the mesh edges

The trained networks are read from models/<formulation>/<material model>.pt (made by the scripts in training/).
"""
import os
import time
from collections import defaultdict

import numpy as np
import torch

from .fem import FEM
from .geometry import DTYPE, lame, polar_rotation, signed_volumes, trace, mtm
from .linear import LinearNetwork, LinearNeuralElements
from .mass_spring import MassSpring
from .materials import Material
from .nonlinear import NonlinearNetwork, NonlinearNeuralElements
from .scenes import SCENES, Boundary
from .solver import INTEGRATORS

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
FORCE_MODELS = {"ne_nonlinear": "Neural Elements, nonlinear formulation",
                "ne_linear": "Neural Elements, linear formulation",
                "fem": "FEM", "mass_spring": "Mass-spring"}
GRAVITY = 9.81


def network_path(formulation, material_model, models_dir=MODELS_DIR):
    return os.path.join(models_dir, formulation, f"{material_model}.pt")


def load_network(formulation, material_model, models_dir=MODELS_DIR):
    path = network_path(formulation, material_model, models_dir)
    if not os.path.exists(path):
        script = "train_linear.py" if formulation == "linear" else f"train_nonlinear.py --material {material_model}"
        raise FileNotFoundError(f"no trained network at {path}; run  python training/{script}")
    return (LinearNetwork if formulation == "linear" else NonlinearNetwork).load(path)


def make_model(name, rest, elems, material, device="cpu", dtype=DTYPE, models_dir=MODELS_DIR):
    if name == "fem":
        return FEM(rest, elems, material, device, dtype)
    if name == "mass_spring":
        return MassSpring(rest, elems, material, device, dtype)
    if name == "ne_linear":
        if material.model != "linear":
            raise ValueError("the linear formulation carries linear materials only; use ne_nonlinear")
        return LinearNeuralElements(rest, elems, material, load_network("linear", "linear", models_dir), device, dtype)
    if name == "ne_nonlinear":
        net = load_network("nonlinear", material.model, models_dir)
        return NonlinearNeuralElements(rest, elems, material, net, device, dtype)
    raise ValueError(f"unknown force model {name!r}; choose one of {list(FORCE_MODELS)}")


class Simulation:
    def __init__(self, scene="cantilever", model="ne_nonlinear", material=None, integrator="newmark", refine=1,
                 device="cpu", dtype=None, models_dir=MODELS_DIR):
        base = SCENES[scene] if isinstance(scene, str) else scene
        self.scene, self.mass = base.refined(refine)                        # mass: nodal mass, M = C = mass I
        self.device = torch.device(device)
        self.dtype = dtype or (torch.float64 if self.device.type == "cpu" else torch.float32)
        m = material or Material()
        self.material = Material(m.model, m.E if m.E is not None else base.E, m.nu if m.nu is not None else base.nu,
                                 m.kappa)
        self.rest, self.elems = self.scene.mesh()                           # on the CPU, double precision
        self.N, self.dt = self.rest.shape[0], self.scene.dt
        self.boundary = Boundary(self.rest, self.scene)
        self.free = (~self.boundary.fixed)[:, None].expand(self.N, 3).to(self.device, self.dtype)
        self.model_name = model
        self.model = make_model(model, self.rest, self.elems, self.material, self.device, self.dtype, models_dir)
        self.integrator = INTEGRATORS[integrator]() if isinstance(integrator, str) else integrator
        self.probe = Probe(self.rest, self.elems, self.boundary)
        self.reset()

    def reset(self):
        z = lambda: torch.zeros(self.N, 3, dtype=self.dtype, device=self.device)
        self.disp, self.vel, self.acc, self.r_internal = z(), z(), z(), z()
        self.frame, self.time = 0, 0.0
        self.history = defaultdict(list)
        self.last = {}

    def external(self):
        f = torch.zeros(self.N, 3, dtype=self.dtype, device=self.device)
        if self.scene.gravity:
            f[:, 1] = -GRAVITY * self.mass
        return f

    def step(self):
        self.constraint = self.boundary.constraint(self.frame).to(self.device, self.dtype)
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = self.integrator.step(self)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        stats["ms"] = 1000 * (time.perf_counter() - t0)
        self.frame += 1
        self.time += self.dt
        m = self.probe.measure(self)
        m.update(stats)
        for k, v in m.items():
            self.history[k].append(v)
        self.last = m
        return m

    def run(self, frames=None):
        for _ in range(frames or self.scene.frames):
            self.step()
            if not torch.isfinite(self.disp).all():
                break
        return self.history

    def set_material(self, E=None, nu=None, kappa=None, mask=None):
        """Change the material while the simulation runs (per element with `mask`); nothing global is rebuilt."""
        self.model.set_material(E=E, nu=nu, kappa=kappa, mask=mask)

    def positions(self):
        """Current node positions (N, 3) as a float32 numpy array (for drawing)."""
        return (self.rest + self.disp.to("cpu", DTYPE)).numpy().astype(np.float32)

    @torch.no_grad()
    def von_mises(self):
        """Nodal von Mises stress (N,) of the element deformations, computed the same way for every force model:
        linear elasticity on each element's rotated strain, averaged at the nodes."""
        md = self.model
        F = md.deformation_gradient(self.disp) if hasattr(md, "deformation_gradient") else None
        if F is None:
            return np.zeros(self.N)
        U = mtm(polar_rotation(F), F)
        eps = 0.5 * (U + U.transpose(1, 2)) - torch.eye(3, dtype=F.dtype, device=F.device)
        lam, mu = lame(md.nu)
        s = md.E[:, None, None] * (lam[:, None, None] * trace(eps)[:, None, None] * torch.eye(3, dtype=F.dtype, device=F.device)
                                   + 2 * mu[:, None, None] * eps)
        dev = s - trace(s)[:, None, None] / 3 * torch.eye(3, dtype=F.dtype, device=F.device)
        vm = torch.sqrt(1.5 * (dev * dev).sum((1, 2))).to("cpu", DTYPE)
        total = torch.zeros(self.N, dtype=DTYPE).index_add_(0, self.elems.reshape(-1), vm.repeat_interleave(4))
        count = torch.zeros(self.N, dtype=DTYPE).index_add_(0, self.elems.reshape(-1), torch.ones(self.elems.numel(), dtype=DTYPE))
        return (total / count).numpy()


class Probe:
    """What is recorded every frame: tip deflection (cantilever), axial and lateral strain between two gauge sections,
    end force and torque at the driven end, volume change and inverted elements."""

    def __init__(self, rest, elems, boundary):
        x = rest[:, 0]
        xs = torch.unique(torch.round(x * 1e4) / 1e4)
        L = float(x.max() - x.min())
        near = lambda v: torch.abs(x - xs[torch.argmin(torch.abs(xs - v))]) < 1e-3
        mid = float(x.min()) + L / 2
        self.elems = elems
        self.tip = x > x.max() - 1e-6
        self.mid, self.g1, self.g2 = near(mid), near(mid - L / 4), near(mid + L / 4)
        self.high = boundary.high
        sec = rest[self.mid][:, 1:]
        self.r0 = float(torch.linalg.norm(sec - sec.mean(dim=0), dim=1).mean())
        self.gauge0 = float(rest[self.g2, 0].mean() - rest[self.g1, 0].mean())
        self.V0 = float(signed_volumes(rest, elems).sum())

    def measure(self, sim):
        d = sim.disp.to("cpu", DTYPE)
        p = sim.rest + d
        sec = p[self.mid][:, 1:]
        vol = signed_volumes(p, self.elems)
        out = dict(time=sim.time, tip=float(d[self.tip, 1].mean()),
                   axial=float(p[self.g2, 0].mean() - p[self.g1, 0].mean()) / self.gauge0 - 1,
                   lateral=float(torch.linalg.norm(sec - sec.mean(dim=0), dim=1).mean()) / self.r0 - 1,
                   volume=float(vol.sum()) / self.V0, inverted=int((vol <= 0).sum()),
                   drive=sim.boundary.value(sim.frame - 1))
        if self.high.any():
            r = sim.r_internal.to("cpu", DTYPE)[self.high]
            q = p[self.high]
            out["force"] = float(r[:, 0].sum())
            out["torque"] = float((q[:, 1] * r[:, 2] - q[:, 2] * r[:, 1]).sum())
        return out
