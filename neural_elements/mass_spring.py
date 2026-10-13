"""Mass-spring system on the edges of the tetrahedral mesh: the simplest deformable model, for comparison.

Every mesh edge is a linear spring with the Van Gelder stiffness k = E sum(V) / l0^2 (the sum over the elements that
share the edge). Springs have no Poisson's ratio and do not resist volume change or shear as a continuum does.
"""
import numpy as np
import torch

from .base import ForceModel
from .geometry import DTYPE, signed_volumes


class MassSpring(ForceModel):
    name = "mass_spring"

    def __init__(self, rest, elems, material, device="cpu", dtype=DTYPE):
        super().__init__(rest, elems, material, device, dtype)
        e = elems.cpu().numpy()
        pairs = np.sort(e[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2), axis=1)
        edges, owner = np.unique(pairs, axis=0, return_inverse=True)
        self.edges = torch.as_tensor(edges, device=self.device)
        self.owner = torch.as_tensor(owner.ravel(), device=self.device)          # edge index of each element edge
        self.vol6 = signed_volumes(rest.to(DTYPE), elems).abs().repeat_interleave(6).to(self.device, dtype)
        r = rest.to(DTYPE)[self.edges.cpu()]
        self.e0 = (r[:, 1] - r[:, 0]).to(self.device, dtype)                    # rest edge vectors
        self.l0 = self.e0.norm(dim=1)
        self.material_changed()

    def material_changed(self):
        w = torch.zeros(len(self.l0), dtype=self.dtype, device=self.device)
        w.index_add_(0, self.owner, self.vol6 * self.E.repeat_interleave(6))
        self.k = w / self.l0 ** 2

    def forces(self, d):
        i, j = self.edges[:, 0], self.edges[:, 1]
        e = self.e0 + (d[j] - d[i])                                             # current edge vectors
        l = e.norm(dim=1).clamp_min(1e-12)
        f = (self.k * (l - self.l0) / l)[:, None] * e
        self._n, self._l = e / l[:, None], l
        out = torch.zeros(self.N, 3, dtype=self.dtype, device=self.device)
        return out.index_add_(0, i, -f).index_add_(0, j, f)

    def element_matrices(self):
        """Per spring: the 6 x 6 tangent [[B, -B], [-B, B]], B = k ((1 - l0/l)(I - n n^T) + n n^T)."""
        n, l = self._n, self._l
        nn_ = n[:, :, None] * n[:, None, :]
        B = self.k[:, None, None] * ((1 - self.l0 / l)[:, None, None] * (self.I - nn_) + nn_)
        K = torch.cat([torch.cat([B, -B], 2), torch.cat([-B, B], 2)], 1)
        return K, self.edges
