"""Neural Elements, linear formulation: single-node load cases (the formulation of the dissertation, optimized).

The reference element
    A regular tetrahedron. Its load case: node A displaced by d along x, nodes B, C, D held; the measured quantity is
    the twelve nodal reaction forces. For a linear material they are proportional to E, so the network works per unit
    E and has two inputs:

        LinearNetwork:  (nu, d) -> 12 reaction forces per unit E          one hidden layer of 8 tanh units

How an element uses it
    1. The element's rotation R is taken out (polar decomposition), leaving its nodal displacements u in its own
       frame: twelve load cases, one per node n and axis k, with displacement u_nk.
    2. Each load case is answered by the network at the strain-equivalent displacement s_n u_nk, with
       s_n = |grad N_n| / |grad N_A| of the reference (so that d means the same strain in any element), minus the
       answer at d = 0 (the offset, computed once per nu).
    3. Transfer to the element, geometry only: the twelve reference forces are reduced by least squares to their
       volumetric and shear parts (a_lambda, a_mu) on the reference element's two geometric bases K_lambda, K_mu, and
       those two numbers scale the element's own bases. Summed over the load cases:

           f = R E (K_lambda v_lambda + K_mu v_mu),     v_nk = a(nu, s_n u_nk) / s_n

       For exact linear data a_lambda = lambda d and a_mu = mu d, and this is exactly linear corotational FEM. The
       reduction is linear, so it is folded into the network's output layer once (12 outputs -> 2).
    4. Tangent: the slopes da/dd of the network, K_t = R E (K_lambda diag(a'_lambda) + K_mu diag(a'_mu)) R^T
       (rotation held fixed).
"""
import torch
from torch import nn

from .base import ForceModel, to_global, to_local
from .geometry import DTYPE, gradient, lame, lame_bases, nodal_forces, polar_rotation, reference_tetrahedron, \
    shape_gradients, trace


# ---------------------------------------------------------------------------------------------------------------------
# reference element data
# ---------------------------------------------------------------------------------------------------------------------
def reference_bases():
    """The reference element's geometric bases for load case (A, x): columns K_lambda[:, 0], K_mu[:, 0] (12 each),
    and its shape gradients."""
    X = reference_tetrahedron()
    G, V = shape_gradients(X[None])
    e = torch.zeros(1, 4, 3, dtype=DTYPE)
    e[0, 0, 0] = 1.0
    kl, km = lame_bases(e, G, V)
    return kl.reshape(12), km.reshape(12), G[0]


def reference_forces(nu, d):
    """Reaction forces (n, 12) per unit E of the reference element, node A displaced by d along x (linear material)."""
    kl, km, _ = reference_bases()
    lam, mu = lame(torch.as_tensor(nu, dtype=DTYPE))
    return d[:, None] * (lam[:, None] * kl + mu[:, None] * km)


def training_data(nu=(0.0, 0.45), d=(-1.0, 1.0), n_nu=41, n_d=41):
    """A regular grid over (nu, d): inputs (n, 2) and target reaction forces (n, 12)."""
    NU, D = torch.meshgrid(torch.linspace(*nu, n_nu, dtype=DTYPE), torch.linspace(*d, n_d, dtype=DTYPE), indexing="ij")
    NU, D = NU.reshape(-1), D.reshape(-1)
    return torch.stack([NU, D], 1), reference_forces(NU, D)


# ---------------------------------------------------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------------------------------------------------
class LinearNetwork(nn.Module):
    """(nu, d) -> 12 reaction forces per unit E: one hidden tanh layer, inputs and outputs scaled to [-1, 1].
    `seeds` copies are held side by side for training (a leading dimension); select(i) keeps one."""

    def __init__(self, hidden=8, seeds=1, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        r = lambda a, *s: (torch.rand(*s, generator=g, dtype=DTYPE) * 2 - 1) * a
        self.W1 = nn.Parameter(r(1.0, seeds, 2, hidden))
        self.b1 = nn.Parameter(r(0.5, seeds, 1, hidden))
        self.W2 = nn.Parameter(r(hidden ** -0.5, seeds, hidden, 12))
        self.b2 = nn.Parameter(r(0.1, seeds, 1, 12))
        for name, n in (("xlo", 2), ("xhi", 2), ("ylo", 12), ("yhi", 12)):
            self.register_buffer(name, torch.zeros(n, dtype=DTYPE))
        self.meta = {}

    def normalized(self, X):
        """Normalized inputs (n, 2) -> normalized outputs (seeds, n, 12)."""
        return torch.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2

    def scale_inputs(self, x):
        return 2 * (x - self.xlo) / (self.xhi - self.xlo) - 1

    def forward(self, nu, d):
        """Reaction forces per unit E (n, 12) of the first copy."""
        y = self.normalized(self.scale_inputs(torch.stack([nu, d], 1)))[0]
        return (y + 1) * (self.yhi - self.ylo) / 2 + self.ylo

    def select(self, i):
        out = LinearNetwork(self.W1.shape[2])
        out.load_state_dict({k: (v[i:i + 1] if k in ("W1", "b1", "W2", "b2") else v) for k, v in self.state_dict().items()})
        return out

    def save(self, path, **meta):
        torch.save({"state": self.state_dict(), "hidden": self.W1.shape[2], "meta": meta}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        net = LinearNetwork(ck["hidden"])
        net.load_state_dict(ck["state"])
        net.meta = ck.get("meta", {})
        return net


# ---------------------------------------------------------------------------------------------------------------------
# the elements
# ---------------------------------------------------------------------------------------------------------------------
class LinearNeuralElements(ForceModel):
    name = "ne_linear"

    def __init__(self, rest, elems, material, network, device="cpu", dtype=DTYPE):
        super().__init__(rest, elems, material, device, dtype)
        kl, km, Gref = reference_bases()
        P0 = torch.linalg.pinv(torch.stack([kl, km], 1))                     # (2, 12): least-squares reduction
        cast = lambda t: t.detach().to(self.device, dtype)
        net = network
        # the network with its output scaling and the reduction folded in: amplitudes a = tanh(z) @ Wa (+ constant)
        self.W1, self.b1 = cast(net.W1[0]), cast(net.b1[0])
        self.Wa = cast((net.W2[0] * (net.yhi - net.ylo) / 2) @ P0.T)           # (H, 2)
        self.xlo, self.xhi = cast(net.xlo), cast(net.xhi)
        # strain normalization s_n = |grad N_n| / |grad N_A| of the reference
        self.s = (self.G.norm(dim=2) / float(Gref[0].norm()))                 # (Ne, 4)
        self.material_changed()

    def _hidden(self, nu, d):
        """tanh units for inputs nu (Ne,) and d (Ne, ...) -> (Ne, ..., H), and dz/dd (H,)."""
        X_nu = 2 * (nu - self.xlo[0]) / (self.xhi[0] - self.xlo[0]) - 1
        c = 2 / (self.xhi[1] - self.xlo[1])
        X_d = c * (d - self.xlo[1]) - 1
        shape = (self.Ne,) + (1,) * (d.dim() - 1) + (-1,)
        z = (X_nu[:, None] * self.W1[0] + self.b1[0]).view(shape) + X_d[..., None] * self.W1[1]
        return torch.tanh(z), c * self.W1[1]

    def material_changed(self):
        h, _ = self._hidden(self.nu, torch.zeros(self.Ne, dtype=self.dtype, device=self.device))
        self.a0 = h @ self.Wa                                                 # offsets (Ne, 2), per unit E

    def element_forces(self, d):
        R = polar_rotation(self.deformation_gradient(d))
        u = self.rotated_displacements(d, R)                                   # (Ne, 4, 3): the 12 load cases
        s = self.s[:, :, None]
        h, dz = self._hidden(self.nu, s * u)                                   # (Ne, 4, 3, H)
        a = h @ self.Wa - self.a0[:, None, None, :]                            # (Ne, 4, 3, 2)
        self._R, self._slope = R, ((1 - h * h) * dz) @ self.Wa                 # da/dd, for the tangent
        f = self._apply(a[..., 0] / s, a[..., 1] / s)
        return to_global(R, self.E[:, None, None] * f)

    def _apply(self, vl, vm):
        """K_lambda v_lambda + K_mu v_mu (nodal vectors (Ne, ..., 4, 3)), from the element's shape gradients."""
        extra = vl.dim() - 3
        G, V = self.G.view(self.Ne, *([1] * extra), 4, 3), self.V.view(self.Ne, *([1] * extra))
        Hl, Hm = gradient(vl, G), gradient(vm, G)
        sigma = trace(Hl)[..., None, None] * self.I + Hm + Hm.transpose(-1, -2)
        return nodal_forces(sigma, G, V)

    def dforces(self, D):
        local = to_local(self._R, D)                                           # (Ne, B, 4, 3)
        sl = self._slope[:, None]
        df = self._apply(sl[..., 0] * local, sl[..., 1] * local)
        return to_global(self._R, self.E[:, None, None, None] * df)
