"""Finite elements (linear tetrahedra) of the material models: the references Neural Elements are compared with.

    linear       linear corotational FEM: each element's rotation R is taken out (polar decomposition of F), the
                 linear element stiffness K0 = E (lambda K_lambda + mu K_mu) acts on the rotated displacements, and the
                 forces are rotated back:  f = R K0 u.  Tangent with the rotation held fixed: R K0 R^T.
    neohookean   nonlinear FEM: f_i = V E P(F) G_i with the Neo-Hookean stress, exact tangent.
    stvk         nonlinear FEM: the same with the St. Venant-Kirchhoff stress, exact tangent.
    blend        (1 - kappa) linear + kappa Neo-Hookean, per element (the forces of an energy blend are the blend of the
                 forces).
"""
from .base import ForceModel, to_global, to_local
from .geometry import DTYPE, gradient, lame, lame_bases, nodal_forces, polar_rotation
from .materials import neohookean_dstress, neohookean_stress, stvk_dstress, stvk_stress


class FEM(ForceModel):
    name = "fem"

    def __init__(self, rest, elems, material, device="cpu", dtype=DTYPE):
        super().__init__(rest, elems, material, device, dtype)
        self.stvk = material.model == "stvk"

    def element_forces(self, d):
        F = self.deformation_gradient(d)
        if self.stvk:
            self._F = F
            return nodal_forces(self.E[:, None, None] * stvk_stress(F, self.nu), self.G, self.V)
        k = self.kappa[:, None, None]
        f = 0.0
        self._F, self._R = F, None
        if bool((self.kappa < 1).any()):
            R = polar_rotation(F)
            u = self.rotated_displacements(d, R)
            self._R = R
            f = f + (1 - k) * to_global(R, self._linear(u))
        if bool((self.kappa > 0).any()):
            P = self.E[:, None, None] * neohookean_stress(F, self.nu)
            f = f + k * nodal_forces(P, self.G, self.V)
        return f

    def _linear(self, u):
        """K0 u = E (lambda K_lambda u + mu K_mu u) for nodal vectors u (Ne, ..., 4, 3), in the element frame."""
        lam, mu = lame(self.nu)
        shape = (self.Ne,) + (1,) * (u.dim() - 1)
        G, V = self.G.view(self.Ne, *([1] * (u.dim() - 3)), 4, 3), self.V.view(self.Ne, *([1] * (u.dim() - 3)))
        fl, fm = lame_bases(u, G, V)
        return self.E.view(shape) * (lam.view(shape) * fl + mu.view(shape) * fm)

    def dforces(self, D):
        if self.stvk:
            dF = gradient(D, self.G[:, None])
            dP = self.E[:, None, None, None] * stvk_dstress(self._F[:, None], self.nu[:, None], dF)
            return nodal_forces(dP, self.G[:, None], self.V[:, None])
        k = self.kappa[:, None, None, None]
        df = 0.0
        if self._R is not None:
            df = df + (1 - k) * to_global(self._R, self._linear(to_local(self._R, D)))
        if bool((self.kappa > 0).any()):
            dF = gradient(D, self.G[:, None])
            dP = self.E[:, None, None, None] * neohookean_dstress(self._F[:, None], self.nu[:, None], dF)
            df = df + k * nodal_forces(dP, self.G[:, None], self.V[:, None])
        return df
