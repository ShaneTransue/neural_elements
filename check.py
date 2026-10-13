"""Self-checks of the library: derivatives, consistency of the models, and the trained networks against FEM.

    python check.py              (about 20 s; adds GPU checks when CUDA is available)
"""
import sys

import torch

from neural_elements import Material
from neural_elements.fem import FEM
from neural_elements.geometry import nodal_forces
from neural_elements.mass_spring import MassSpring
from neural_elements.materials import linear_stress
from neural_elements.scenes import SCENES, beam
from neural_elements.simulation import make_model

ok = True


def report(name, value, tol):
    global ok
    passed = value < tol
    ok &= passed
    print(f"  {'pass' if passed else 'FAIL'}  {name}: {value:.2e}  (tolerance {tol:g})")


def assembled_product(model, p):
    """K p with the element matrices of the model's last state."""
    K, conn = model.element_matrices()
    dof = (3 * conn[:, :, None] + torch.arange(3, device=conn.device)).reshape(len(conn), -1)
    y = torch.bmm(K, p.reshape(-1)[dof][:, :, None])[:, :, 0]
    return torch.zeros(p.numel(), dtype=p.dtype, device=p.device).index_add_(0, dof.reshape(-1), y.reshape(-1)), K


torch.manual_seed(0)
rest, elems = beam((3, 2, 2), (3.0, 2.0, 2.0))
d = 0.15 * torch.randn_like(rest)

print("1. exact tangents: element matrices against finite differences of the forces")
cases = [("FEM, Neo-Hookean", FEM(rest, elems, Material("neohookean", 100.0, 0.3))),
         ("FEM, St. Venant-Kirchhoff", FEM(rest, elems, Material("stvk", 100.0, 0.3))),
         ("mass-spring", MassSpring(rest, elems, Material("linear", 100.0, 0.3)))]
for m in ("linear", "neohookean", "stvk", "blend"):
    cases.append((f"Neural Elements, nonlinear formulation, {m}",
                  make_model("ne_nonlinear", rest, elems, Material(m, 100.0, 0.3, 0.4))))
for name, model in cases:
    model.forces(d)
    p = torch.randn_like(rest)
    Kp, K = assembled_product(model, p)
    fd = ((model.forces(d + 1e-6 * p) - model.forces(d - 1e-6 * p)) / 2e-6).reshape(-1)
    report(f"{name}: tangent", float((Kp - fd).norm() / fd.norm()), 1e-6)
    report(f"{name}: tangent symmetry", float((K - K.mT).norm() / K.norm()), 1e-12)

print("2. linear corotational FEM equals f_i = V P(F) G_i with the rotated linear stress")
fem = FEM(rest, elems, Material("linear", 100.0, 0.3))
f = fem.element_forces(d)
P = 100.0 * linear_stress(fem.deformation_gradient(d), fem.nu)
report("forces", float((f - nodal_forces(P, fem.G, fem.V)).norm() / f.norm()), 1e-12)

print("3. Neural Elements against the FEM of the same material: isolated elements of the scene meshes, deformed")
print("   by random rotations and stretches up to 30% (median relative force error over elements)")
g = torch.Generator().manual_seed(1)
Xs = []
for scene in SCENES.values():
    r, e = scene.mesh()
    Xs.append(r[e[torch.randperm(len(e), generator=g)[:60]]])
X = torch.cat(Xs)
iso_rest, iso_elems = X.reshape(-1, 3), torch.arange(4 * len(X)).reshape(-1, 4)
Q, _ = torch.linalg.qr(torch.randn(len(X), 3, 3, generator=g, dtype=X.dtype))
R, _ = torch.linalg.qr(torch.randn(len(X), 3, 3, generator=g, dtype=X.dtype))
R = R * torch.sign(torch.linalg.det(R))[:, None, None]
for strain in (0.01, 0.3):
    S = torch.diag_embed(1 + strain * (2 * torch.rand(len(X), 3, generator=g, dtype=X.dtype) - 1))
    F = R @ Q @ S @ Q.transpose(1, 2)
    Xc = X.mean(1, keepdim=True)
    disp = ((Xc + torch.einsum("eab,eib->eia", F, X - Xc)) - X).reshape(-1, 3)
    for mat, model in (("linear", "ne_linear"), ("linear", "ne_nonlinear"), ("neohookean", "ne_nonlinear"),
                       ("stvk", "ne_nonlinear"), ("blend", "ne_nonlinear")):
        material = Material(mat, 1.0, 0.3, 0.5)
        ref = make_model("fem", iso_rest, iso_elems, material).forces(disp).view(-1, 12)
        out = make_model(model, iso_rest, iso_elems, material).forces(disp).view(-1, 12)
        err = 100 * float(((out - ref).norm(dim=1) / ref.norm(dim=1)).median())
        report(f"{model:12s} {mat:10s} at {100 * strain:4.0f}% strain [%]", err, 1.5)

if torch.cuda.is_available():
    print("4. GPU (single precision) against CPU (double precision): forces of every model on the twist mesh")
    r, e = SCENES["twist"].mesh()
    dd = 0.2 * torch.randn_like(r)
    for model, mat in (("fem", "linear"), ("fem", "neohookean"), ("fem", "stvk"), ("ne_linear", "linear"),
                       ("ne_nonlinear", "neohookean"), ("ne_nonlinear", "stvk"), ("ne_nonlinear", "blend"),
                       ("mass_spring", "linear")):
        material = Material(mat, 5900.0, 0.3, 0.5)
        cpu = make_model(model, r, e, material).forces(dd)
        gpu = make_model(model, r, e, material, "cuda", torch.float32).forces(dd.to("cuda", torch.float32))
        report(f"{model:12s} {mat:10s}", float((gpu.double().cpu() - cpu).norm() / cpu.norm()), 1e-4)

print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
