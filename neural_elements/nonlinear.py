"""Neural Elements, nonlinear formulation: one deformation load case per element, learned as an energy.

The reference element
    The same regular tetrahedron, but its load case is a whole deformation: all four nodes move, as given by a
    deformation gradient F. Its reaction forces are the gradient of its strain energy, f_i = V P G_i with
    P = dW/dF. The network learns W per unit E:

        NonlinearNetwork:  x = (nu, [kappa], I1 - 3, I2 - 3, ln J) -> W        one hidden layer of tanh units

    I1, I2 are invariants of C = F^T F and J = det F. C holds the element's six edge lengths, so no rotation has to
    be extracted, and W cannot depend on the element's orientation or on the direction it is stretched in
    (objective and isotropic by construction). The network is trained on reaction forces only; W itself is never
    observed.

How an element uses it
    F = I + sum_i d_i (x) G_i from its nodal displacements; the stress follows from the network's own derivative,

        P = dW/dI1 2F + dW/dI2 2(I1 F - F C) + (dW/dlnJ - n0) F^-T

    and is transferred by the element's geometry: f_i = V E P G_i (exact for any element shape). n0 removes the
    network's stress at rest; it depends on the material inputs only and is computed once per material change (the
    offset of the linear formulation). The tangent is the exact second derivative of W: symmetric, and the forces
    conserve energy.

Material control: the inputs after nu are the material parameters of the network's model (kappa for the blend).
"""
import torch
from torch import nn

from .base import ForceModel
from .geometry import DTYPE, cofactor, gradient, mm, mtm, nodal_forces, reference_tetrahedron, shape_gradients, \
    trace
from .materials import stress

PARAMETERS = {"linear": (), "neohookean": (), "stvk": (), "blend": ("kappa",)}


def invariants(F):
    """x = (I1 - 3, I2 - 3, ln J) (..., 3), and C, I1, F^-T for the stress."""
    C = mtm(F, F)
    I1 = trace(C)
    I2 = 0.5 * (I1 * I1 - (C * C).sum((-2, -1)))
    cof, J = cofactor(F)
    J = J.clamp_min(1e-4)                                   # an inverted element is pushed back instead of NaN
    return torch.stack([I1 - 3, I2 - 3, torch.log(J)], -1), C, I1, cof / J[..., None, None]


def stress_from_derivatives(g, F, C, I1, FinvT):
    """P = g1 dI1/dF + g2 dI2/dF + g3 dlnJ/dF for g = dW/dx (..., 3)."""
    c = lambda i: g[..., i, None, None]
    return 2 * c(0) * F + 2 * c(1) * (I1[..., None, None] * F - mm(F, C)) + c(2) * FinvT


# ---------------------------------------------------------------------------------------------------------------------
# reference element data
# ---------------------------------------------------------------------------------------------------------------------
REFERENCE_G, REFERENCE_V = (t[0] for t in shape_gradients(reference_tetrahedron()[None]))


def reference_forces(P):
    """Reaction forces (..., 12) of the reference element under the stress P (..., 3, 3)."""
    return nodal_forces(P, REFERENCE_G, REFERENCE_V).flatten(-2)


def training_data(model, n=20000, nu=(0.0, 0.45), stretch=(0.65, 1.45), seed=0):
    """Reference load cases: nu (and kappa) uniform, a stretch F = Q diag(l1, l2, l3) Q^T with random principal
    stretches and directions. Returns material inputs z (n, 1 or 2), F (n, 3, 3), reaction forces (n, 12)."""
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, generator=g, dtype=DTYPE)
    nus = u(*nu, n)
    Q, _ = torch.linalg.qr(torch.randn(n, 3, 3, generator=g, dtype=DTYPE))
    F = Q @ torch.diag_embed(u(*stretch, n, 3)) @ Q.transpose(1, 2)
    kappa = u(0, 1, n)                                            # used by the blend only
    z = torch.stack([nus, kappa], 1) if PARAMETERS[model] else nus[:, None]
    return z, F, reference_forces(stress(model, F, nus, kappa))


# ---------------------------------------------------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------------------------------------------------
class NonlinearNetwork(nn.Module):
    """W = w2 . tanh(W1 X + b1), X = x scaled to [-1, 1]; x = (material inputs z, I1 - 3, I2 - 3, ln J).
    `seeds` copies side by side for training; select(i) keeps one."""

    def __init__(self, model="neohookean", hidden=16, seeds=1, seed=0):
        super().__init__()
        self.model = model
        nin = 1 + len(PARAMETERS[model]) + 3
        g = torch.Generator().manual_seed(seed)
        r = lambda a, *s: (torch.rand(*s, generator=g, dtype=DTYPE) * 2 - 1) * a
        self.W1 = nn.Parameter(r(1.0, seeds, nin, hidden))
        self.b1 = nn.Parameter(r(0.5, seeds, 1, hidden))
        self.w2 = nn.Parameter(r(hidden ** -0.5, seeds, 1, hidden))
        self.register_buffer("xlo", -torch.ones(nin, dtype=DTYPE))
        self.register_buffer("xhi", torch.ones(nin, dtype=DTYPE))
        self.meta = {}

    def derivatives(self, z, x, hessian=False):
        """dW/dx (seeds, n, 3) for the three invariant inputs, and d2W/dx2 (seeds, n, 3, 3) if `hessian`."""
        s = 2 / (self.xhi - self.xlo)
        X = (torch.cat([z, x], -1) - self.xlo) * s - 1
        h = torch.tanh(X @ self.W1 + self.b1)                                  # (seeds, n, H)
        Wx = self.W1[:, -3:, :] * s[-3:, None]                                 # (seeds, 3, H): d z_h / d x
        g = ((1 - h * h) * self.w2) @ Wx.transpose(1, 2)
        if not hessian:
            return g
        q = -2 * h * (1 - h * h) * self.w2                                    # d2 tanh, times w2
        return g, torch.einsum("skh,snh,slh->snkl", Wx, q, Wx)

    def stress(self, z, F):
        """P per unit E (seeds, n, 3, 3) with the stress at rest removed (used for training)."""
        x, C, I1, FinvT = invariants(F)
        g = self.derivatives(z, x)
        g0 = self.derivatives(z, torch.zeros_like(x))
        g = g - torch.stack([torch.zeros_like(g0[..., 0])] * 2 + [rest_stress(g0)], -1)
        return stress_from_derivatives(g, F, C, I1, FinvT)

    def select(self, i):
        out = NonlinearNetwork(self.model, self.W1.shape[2])
        out.load_state_dict({k: (v[i:i + 1] if k in ("W1", "b1", "w2") else v) for k, v in self.state_dict().items()})
        return out

    def save(self, path, **meta):
        torch.save({"state": self.state_dict(), "hidden": self.W1.shape[2], "model": self.model, "meta": meta}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        net = NonlinearNetwork(ck["model"], ck["hidden"])
        net.load_state_dict(ck["state"])
        net.meta = ck.get("meta", {})
        return net


def rest_stress(g0):
    """At F = I the stress is (2 W_I1 + 4 W_I2 + W_lnJ) I; the term -n0 ln J in the energy removes it."""
    return 2 * g0[..., 0] + 4 * g0[..., 1] + g0[..., 2]


# ---------------------------------------------------------------------------------------------------------------------
# the elements
# ---------------------------------------------------------------------------------------------------------------------
class NonlinearNeuralElements(ForceModel):
    name = "ne_nonlinear"

    def __init__(self, rest, elems, material, network, device="cpu", dtype=DTYPE):
        super().__init__(rest, elems, material, device, dtype)
        self.net = network.to(self.device, dtype)
        self.net.requires_grad_(False)
        self.material_changed()

    def _inputs(self):
        """Material inputs z (Ne, 1 or 2): nu, and kappa for the blend network."""
        return torch.stack([self.nu, self.kappa], 1) if PARAMETERS[self.net.model] else self.nu[:, None]

    def material_changed(self):
        z = self._inputs()
        self.n0 = rest_stress(self.net.derivatives(z, torch.zeros(self.Ne, 3, dtype=self.dtype, device=self.device))[0])

    def element_forces(self, d):
        F = self.deformation_gradient(d)
        x, C, I1, FinvT = invariants(F)
        g, H = self.net.derivatives(self._inputs(), x, hessian=True)
        g = torch.cat([g[0, :, :2], g[0, :, 2:] - self.n0[:, None]], 1)
        self._state = (F, C, I1, FinvT, g, H[0])
        P = self.E[:, None, None] * stress_from_derivatives(g, F, C, I1, FinvT)
        return nodal_forces(P, self.G, self.V)

    def dforces(self, D):
        """Exact: dP = sum_k g_k d(dx_k/dF)[dF] + sum_kl H_kl (dx_l/dF : dF) dx_k/dF."""
        F, C, I1, FinvT, g, H = (t[:, None] for t in self._state)               # broadcast over the B directions
        dF = gradient(D, self.G[:, None])
        dI2 = I1[..., None, None] * F - mm(F, C)                               # dI2/dF / 2
        dx = torch.stack([2 * (F * dF).sum((-2, -1)), 2 * (dI2 * dF).sum((-2, -1)), (FinvT * dF).sum((-2, -1))], -1)
        dg = (H * dx[..., None, :]).sum(-1)                                    # (Ne, B, 3)
        dC = mtm(dF, F) + mtm(F, dF)
        c = lambda t, i: t[..., i, None, None]
        dP = (2 * c(dg, 0) * F + 2 * c(g, 0) * dF
              + 2 * c(dg, 1) * dI2 + 2 * c(g, 1) * (c(dx, 0) * F + I1[..., None, None] * dF - mm(dF, C) - mm(F, dC))
              + c(dg, 2) * FinvT - c(g, 2) * mm(mm(FinvT, dF.transpose(-1, -2)), FinvT))
        return nodal_forces(self.E[:, None, None, None] * dP, self.G[:, None], self.V[:, None])
