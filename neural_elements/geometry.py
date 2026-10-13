"""Geometry of linear tetrahedra, written as element-wise tensor operations.

Every function works on a batch of elements at once (leading dimensions ...), on the CPU or the GPU. Short
contractions (over 3 axes or 4 nodes) are written as broadcast products summed over the short index instead of
batched matrix products: on the GPU, batched products of tiny matrices are slow, element-wise kernels are fast.

Notation for one element: rest nodes X_i, displacements d_i (i = 0..3), shape-function gradients G_i = grad N_i.
    deformation gradient   F = I + sum_i d_i (x) G_i
    nodal forces of a stress P (first Piola-Kirchhoff):   f_i = V P G_i
"""
import numpy as np
import torch

DTYPE = torch.float64


def mm(A, B):
    """A @ B for stacks of 3x3 matrices, element-wise."""
    return (A[..., :, :, None] * B[..., None, :, :]).sum(-2)


def mtm(A, B):
    """A^T @ B for stacks of 3x3 matrices, element-wise."""
    return (A[..., :, :, None] * B[..., :, None, :]).sum(-3)


def trace(A):
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)


def shape_gradients(X):
    """X (Ne, 4, 3) rest positions -> G (Ne, 4, 3) gradients of the shape functions, V (Ne,) volumes."""
    Dm = (X[:, 1:] - X[:, :1]).transpose(1, 2)          # edge vectors from node 0, as columns
    Gi = torch.linalg.inv(Dm)                           # rows: gradients of N_1, N_2, N_3
    G = torch.cat([-Gi.sum(dim=1, keepdim=True), Gi], dim=1)
    return G, torch.linalg.det(Dm).abs() / 6.0


def gradient(d, G):
    """sum_i d_i (x) G_i: the displacement gradient (..., 3, 3) of nodal values d (..., 4, 3)."""
    return (d[..., :, :, None] * G[..., :, None, :]).sum(-3)


def nodal_forces(P, G, V):
    """f_i = V P G_i (..., 4, 3) for stresses P (..., 3, 3)."""
    return V[..., None, None] * (P[..., None, :, :] * G[..., :, None, :]).sum(-1)


def cofactor(F):
    """cof(F) = det(F) F^-T and det(F), from cross products of the columns (no matrix inverse)."""
    c0, c1, c2 = F[..., :, 0], F[..., :, 1], F[..., :, 2]
    k0 = torch.linalg.cross(c1, c2, dim=-1)
    cof = torch.stack([k0, torch.linalg.cross(c2, c0, dim=-1), torch.linalg.cross(c0, c1, dim=-1)], dim=-1)
    return cof, (c0 * k0).sum(-1)


def polar_rotation(F, iterations=7):
    """Rotation R of the polar decomposition F = R U, by the scaled Newton iteration X <- (g X + X^-T / g) / 2,
    g = |det X|^(-1/3), which converges quadratically for det F > 0. Inverted elements (det F <= 0, rare) take the
    closest proper rotation from an SVD of only those elements."""
    X = F
    _, det0 = cofactor(F)
    for _ in range(iterations):
        cof, det = cofactor(X)
        g = det.abs().clamp_min(1e-30) ** (-1.0 / 3.0)
        X = 0.5 * (g[..., None, None] * X + cof / (g * det)[..., None, None])
    bad = det0 <= 0
    if bool(bad.any()):
        U, _, Vh = torch.linalg.svd(F[bad])
        s = torch.sign(torch.linalg.det(U @ Vh))
        U = torch.cat([U[..., :2], U[..., 2:] * s[:, None, None]], dim=-1)
        X = X.clone()
        X[bad] = U @ Vh
    return X


def lame(nu):
    """Lame constants per unit Young's modulus: lambda / E and mu / E."""
    return nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))


def lame_bases(v, G, V):
    """The two geometric stiffness bases of an element applied to nodal values v (..., 4, 3):

        K_lambda v:  f_i = V tr(H) G_i            (volumetric)
        K_mu v:      f_i = V (H + H^T) G_i        (shear),          H = sum_j v_j (x) G_j

    Any isotropic linear element is K = lambda K_lambda + mu K_mu. Both bases depend on the element's shape only."""
    H = gradient(v, G)
    I = torch.eye(3, dtype=v.dtype, device=v.device)
    return nodal_forces(trace(H)[..., None, None] * I, G, V), nodal_forces(H + H.transpose(-1, -2), G, V)


def reference_tetrahedron(dtype=DTYPE):
    """The regular reference tetrahedron: alternating corners of a cube, node A at the origin, edge 1.5731 (the
    mean edge of the dissertation's reference, so that a displacement d means the same relative deformation)."""
    cube = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    p = cube * (1.5731321849709865 / (2 * np.sqrt(2)))
    return torch.tensor(p - p[0], dtype=dtype)


def signed_volumes(x, elems):
    p = x[elems]
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    return (torch.linalg.cross(a, b, dim=1) * c).sum(1) / 6.0
