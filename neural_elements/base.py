"""What every force model has in common.

A force model turns nodal displacements into internal nodal forces, and gives its tangent stiffness element by
element. The time integrators only use these two calls:

    forces(d)            d (N, 3) nodal displacements -> internal forces (N, 3); keeps what the tangent needs
    element_matrices()   the tangent at the state of the last forces() call: K (Ne, 3m, 3m) and the element nodes
                         conn (Ne, m) (m = 4 for tetrahedra, 2 for springs)

and the material can be changed per element at any time with set_material(E, nu, kappa, mask).

Tetrahedral models implement element_forces(d) -> (Ne, 4, 3) and dforces(D): the change of the element forces for
nodal displacement changes D (Ne, B, 4, 3). The element matrices then follow from the 12 unit directions.
"""
import torch

from .geometry import DTYPE, shape_gradients


class ForceModel:
    name = "model"

    def __init__(self, rest, elems, material, device="cpu", dtype=DTYPE):
        self.device, self.dtype = torch.device(device), dtype
        self.N, self.Ne = rest.shape[0], elems.shape[0]
        self.elems = elems.to(self.device)
        G, V = shape_gradients(rest.to(DTYPE)[elems])              # rest geometry in double precision, then cast
        Xe = rest.to(DTYPE)[elems]
        self.G, self.V = G.to(self.device, dtype), V.to(self.device, dtype)
        self.Xc = (Xe - Xe.mean(dim=1, keepdim=True)).to(self.device, dtype)    # centered rest nodes per element
        full = lambda v: torch.full((self.Ne,), float(v), dtype=dtype, device=self.device)
        self.E, self.nu, self.kappa = full(material.E), full(material.nu), full(material.nonlinearity)
        self.I = torch.eye(3, dtype=dtype, device=self.device)

    # material ------------------------------------------------------------------------------------------------------
    def set_material(self, E=None, nu=None, kappa=None, mask=None):
        """Change E, nu or kappa of all elements, or of those in `mask` (bool (Ne,)); values may be per element."""
        mask = slice(None) if mask is None else mask.to(self.device)
        for name, v in (("E", E), ("nu", nu), ("kappa", kappa)):
            if v is None:
                continue
            v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
            getattr(self, name)[mask] = v[mask] if v.ndim else v
        self.material_changed()

    def material_changed(self):
        """Recompute what depends on the material only (offsets, constants)."""

    # forces ----------------------------------------------------------------------------------------------------------
    def forces(self, d):
        return self.scatter(self.element_forces(d))

    def element_forces(self, d):
        raise NotImplementedError

    def dforces(self, D):
        raise NotImplementedError

    def scatter(self, f):
        """Sum element nodal forces (Ne, 4, 3) at the nodes -> (N, 3)."""
        out = torch.zeros(self.N, 3, dtype=self.dtype, device=self.device)
        return out.index_add_(0, self.elems.reshape(-1), f.reshape(-1, 3))

    def element_matrices(self):
        """K[e, i, j] = d f_i / d u_j for the 12 nodal degrees of freedom of each element."""
        D = torch.eye(12, dtype=self.dtype, device=self.device).view(1, 12, 4, 3).expand(self.Ne, 12, 4, 3)
        return self.dforces(D).reshape(self.Ne, 12, 12).transpose(1, 2), self.elems

    # helpers ---------------------------------------------------------------------------------------------------------
    def deformation_gradient(self, d):
        """F = I + sum_i d_i (x) G_i per element, from displacements (accurate in single precision too)."""
        de = d[self.elems]
        return self.I + (de[:, :, :, None] * self.G[:, :, None, :]).sum(1)

    def rotated_displacements(self, d, R):
        """Element displacements in the element's rotated frame: u_i = R^T (x_i - x_c) - (X_i - X_c)."""
        de = d[self.elems]
        dc = de - de.mean(dim=1, keepdim=True)
        return to_local(R, dc) + to_local(R, self.Xc) - self.Xc


def to_local(R, v):
    """R^T v_i for nodal vectors v (Ne, ..., 4, 3) and rotations R (Ne, 3, 3)."""
    Rb = R.view(R.shape[0], *([1] * (v.dim() - 2)), 3, 3)
    return (Rb * v[..., :, None]).sum(-2)


def to_global(R, v):
    """R v_i for nodal vectors v (Ne, ..., 4, 3)."""
    Rb = R.view(R.shape[0], *([1] * (v.dim() - 2)), 3, 3)
    return (Rb * v[..., None, :]).sum(-1)
