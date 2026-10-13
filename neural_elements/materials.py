"""The material definition, and the material models that generate training data and drive the FEM references.

A material is what a user sets:

    Material(model="neohookean", E=9500, nu=0.3)
    Material(model="blend", kappa=0.5)            # halfway between linear and Neo-Hookean

    model   "linear"       linear elasticity (in each element's rotated frame, so large rotations are handled)
            "neohookean"   compressible Neo-Hookean, psi = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2
            "stvk"         St. Venant-Kirchhoff, psi = lam/2 tr(E)^2 + mu tr(E^2) with the Green strain
                           E = (F^T F - I)/2: linear elasticity applied to a rotation-free strain. It stiffens in
                           tension and softens in compression; past a stretch of 1/sqrt(3) (about 42% uniaxial
                           compression) it loses stiffness altogether, and an element turned inside out has no energy
                           pushing it back. Use it for moderate compression.
            "blend"        psi = (1 - kappa) psi_linear + kappa psi_neohookean: one parameter from linear (0) to
                           Neo-Hookean (1); every kappa is a valid material with the same small-strain behavior
    E       Young's modulus. Forces scale with E, so it is not a network input (the networks work per unit E).
    nu      Poisson's ratio, a network input (trained range 0 - 0.45).
    kappa   blend parameter, a network input of the blend model.

E and nu left as None take the scene's values. Every model can change E, nu and kappa per element while a
simulation runs; nothing global is rebuilt.

The stress functions below (first Piola-Kirchhoff stress P per unit E) are the ground truth of each model: the
training scripts sample them on the reference element, and the FEM force model evaluates them directly.
"""
from dataclasses import dataclass

import torch

from .geometry import cofactor, lame, mm, mtm, polar_rotation, trace

MODELS = ("linear", "neohookean", "stvk", "blend")


@dataclass
class Material:
    model: str = "linear"
    E: float = None
    nu: float = None
    kappa: float = 0.5

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"unknown material model {self.model!r}; choose one of {MODELS}")

    @property
    def nonlinearity(self):
        """The blend weight of the Neo-Hookean part: 0 for linear, 1 for Neo-Hookean, kappa for the blend (not used
        by St. Venant-Kirchhoff)."""
        return {"linear": 0.0, "neohookean": 1.0, "stvk": 0.0, "blend": self.kappa}[self.model]


def linear_stress(F, nu):
    """Linear elasticity in the rotated frame: F = R U, eps = U - I, P = R (lam tr(eps) I + 2 mu eps)."""
    lam, mu = lame(nu)
    R = polar_rotation(F)
    U = mtm(R, F)
    eps = 0.5 * (U + U.transpose(-1, -2)) - torch.eye(3, dtype=F.dtype, device=F.device)
    T = lam[..., None, None] * trace(eps)[..., None, None] * torch.eye(3, dtype=F.dtype, device=F.device) \
        + 2 * mu[..., None, None] * eps
    return mm(R, T)


def neohookean_parts(F, nu):
    """Neo-Hookean stress P = mu F + (lam ln J - mu) F^-T, with F^-T and ln J. J is kept above a small positive
    value, so that an inverted element (which the model does not define) gets a large restoring stress, not NaN."""
    lam, mu = lame(nu)
    cof, J = cofactor(F)
    J = J.clamp_min(1e-4)
    FinvT, lnJ = cof / J[..., None, None], torch.log(J)
    return mu[..., None, None] * F + (lam * lnJ - mu)[..., None, None] * FinvT, FinvT, lnJ


def neohookean_stress(F, nu):
    return neohookean_parts(F, nu)[0]


def neohookean_dstress(F, nu, dF):
    """The change of the Neo-Hookean stress for a change dF of the deformation gradient:
    dP = mu dF + lam (F^-T : dF) F^-T - (lam ln J - mu) F^-T dF^T F^-T."""
    lam, mu = lame(nu)
    _, FinvT, lnJ = neohookean_parts(F, nu)
    a = (FinvT * dF).sum((-2, -1))
    dFinvT = -mm(mm(FinvT, dF.transpose(-1, -2)), FinvT)
    return (mu[..., None, None] * dF + (lam * a)[..., None, None] * FinvT
            + (lam * lnJ - mu)[..., None, None] * dFinvT)


def stvk_parts(F, nu):
    """St. Venant-Kirchhoff: Green strain E = (F^T F - I)/2, S = lam tr(E) I + 2 mu E, P = F S. Returns P and S."""
    lam, mu = lame(nu)
    I = torch.eye(3, dtype=F.dtype, device=F.device)
    E = 0.5 * (mtm(F, F) - I)
    S = lam[..., None, None] * trace(E)[..., None, None] * I + 2 * mu[..., None, None] * E
    return mm(F, S), S


def stvk_stress(F, nu):
    return stvk_parts(F, nu)[0]


def stvk_dstress(F, nu, dF):
    """dP = dF S + F dS, dS = lam tr(dE) I + 2 mu dE, dE = (dF^T F + F^T dF)/2."""
    lam, mu = lame(nu)
    _, S = stvk_parts(F, nu)
    dE = 0.5 * (mtm(dF, F) + mtm(F, dF))
    I = torch.eye(3, dtype=F.dtype, device=F.device)
    dS = lam[..., None, None] * trace(dE)[..., None, None] * I + 2 * mu[..., None, None] * dE
    return mm(dF, S) + mm(F, dS)


def stress(model, F, nu, kappa=None):
    """P per unit E of a material model (kappa: the blend parameter, per element)."""
    if model == "stvk":
        return stvk_stress(F, nu)
    if model == "linear":
        return linear_stress(F, nu)
    if model == "neohookean":
        return neohookean_stress(F, nu)
    k = torch.as_tensor(kappa, dtype=F.dtype, device=F.device).expand(F.shape[:-2])[..., None, None]
    return (1 - k) * linear_stress(F, nu) + k * neohookean_stress(F, nu)
