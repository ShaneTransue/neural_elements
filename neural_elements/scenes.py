"""Meshes and the four test scenes: cantilever, stretch, compress, twist.

The beams, materials and time steps follow the dissertation application's scenes (unit nodal masses and damping,
one step per frame). The driven end of stretch and compress moves out and back smoothly, d(t) = +-A (1 - cos phase)/2,
and the twist turns the end by A sin(phase); the phase advances 0.02 per frame.
"""
import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import DTYPE

# ---------------------------------------------------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------------------------------------------------
_TET5_EVEN = [[1, 2, 4, 7], [0, 1, 2, 4], [3, 2, 1, 7], [5, 1, 4, 7], [6, 2, 7, 4]]
_TET5_ODD = [[0, 3, 5, 6], [1, 0, 5, 3], [2, 3, 6, 0], [4, 5, 0, 6], [7, 6, 5, 3]]


def beam(cells, size, centered=False):
    """A box of nx * ny * nz cubes, each split into five tetrahedra (alternating, so neighbors match).
    Returns rest positions (N, 3) and elements (Ne, 4), all positively oriented."""
    nx, ny, nz = cells
    xs, ys, zs = (np.linspace(0, s, n + 1) for s, n in zip(size, cells))
    X = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    idx = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)
    tets = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = [idx(i + (a & 1), j + ((a >> 1) & 1), k + ((a >> 2) & 1)) for a in range(8)]
                tets += [[c[a] for a in t] for t in (_TET5_EVEN if (i + j + k) % 2 == 0 else _TET5_ODD)]
    if centered:
        X = X - X.mean(axis=0)
    rest, elems = torch.tensor(X, dtype=DTYPE), torch.tensor(tets)
    p = rest[elems]
    vol = (torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], dim=1) * (p[:, 3] - p[:, 0])).sum(1)
    elems[vol < 0] = elems[vol < 0][:, [0, 2, 1, 3]]
    return rest, elems


def boundary_faces(elems):
    """Outward-facing boundary triangles (Nf, 3): the faces that belong to exactly one element."""
    faces = {}
    for a, b, c, d in elems.tolist():
        for f in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):      # outward for positively oriented elements
            key = tuple(sorted(f))
            faces[key] = None if key in faces else f
    return np.array([f for f in faces.values() if f is not None])


# ---------------------------------------------------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Scene:
    name: str
    cells: tuple
    size: tuple
    centered: bool
    E: float
    dt: float
    frames: int
    nu: float = 0.3
    gravity: bool = False
    clamp: float = 0.05          # fraction of the length held at each constrained end
    drive: str = "none"          # none | pull | push | cycle | twist
    amplitude: float = 0.0
    rate: float = 0.02           # phase advance per frame
    description: str = ""

    def mesh(self):
        return beam(self.cells, self.size, self.centered)

    def refined(self, refine):
        """The scene with `refine` times the cells along each axis, and the nodal mass that keeps the total mass the
        same (masses, damping and weight are per node)."""
        nodes = lambda c: (c[0] + 1) * (c[1] + 1) * (c[2] + 1)
        cells = tuple(c * refine for c in self.cells)
        return dataclasses.replace(self, cells=cells), nodes(self.cells) / nodes(cells)

    @property
    def loading_frames(self):
        """Frames of the loading half of the drive cycle (out to the peak)."""
        return int(math.pi / self.rate)


SCENES = {
    "cantilever": Scene("cantilever", (8, 4, 4), (7.1111, 3.2, 3.2), False, E=9900, dt=1 / 60, frames=900,
                        gravity=True, clamp=0.10, description="clamped at one end, bending under its weight"),
    "stretch": Scene("stretch", (12, 4, 4), (11.0769, 3.2, 3.2), True, E=9500, dt=1 / 60, frames=320, drive="pull",
                     amplitude=3.5, description="one end pulled out by 3.5 (32% strain) and back"),
    # kept below buckling: beyond about 27% axial strain the clamped beam bows sideways
    "compress": Scene("compress", (12, 4, 4), (11.0769, 3.2, 3.2), True, E=9500, dt=1 / 60, frames=320, drive="push",
                      amplitude=2.0, description="one end pushed in by 2.0 (18% strain) and back"),
    "twist": Scene("twist", (14, 6, 6), (11.2, 3.4286, 3.4286), True, E=5900, dt=1 / 30, frames=320, drive="twist",
                   amplitude=math.pi, description="one end turned by up to half a turn each way"),
}


class Boundary:
    """The low end is fixed; the high end is driven (pull, push, twist) or free."""

    def __init__(self, rest, scene):
        x = rest[:, 0]
        L = float(x.max() - x.min())
        self.rest, self.scene = rest, scene
        self.low = x < x.min() + scene.clamp * L + 1e-9
        self.high = (x > x.max() - scene.clamp * L - 1e-9) if scene.drive != "none" else torch.zeros_like(self.low)
        self.fixed = self.low | self.high

    def value(self, frame):
        """Drive value at the end of step `frame`: displacement along x, or rotation angle."""
        ph, A = self.scene.rate * (frame + 1), self.scene.amplitude
        return {"pull": A * (1 - math.cos(ph)) / 2, "push": -A * (1 - math.cos(ph)) / 2,
                "cycle": A * math.sin(ph), "twist": A * math.sin(ph)}.get(self.scene.drive, 0.0)

    def constraint(self, frame):
        """Prescribed displacements (N, 3) of the constrained nodes at the end of step `frame`."""
        c = torch.zeros_like(self.rest)
        v = self.value(frame)
        if self.scene.drive in ("pull", "push", "cycle"):     # cycle: out and in, as the original application
            c[self.high, 0] = v
        elif self.scene.drive == "twist":
            cs, sn = math.cos(v), math.sin(v)
            R = torch.tensor([[1, 0, 0], [0, cs, -sn], [0, sn, cs]], dtype=self.rest.dtype)
            p = self.rest[self.high]
            c[self.high] = p @ R.T - p
        return c
