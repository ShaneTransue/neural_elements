"""Time integration. Every node has the same mass m and damping c = m (the scenes' unit masses and damping).

    Newmark        implicit, average acceleration (beta = 1/4, gamma = 1/2)
    BackwardEuler  implicit, more strongly damped
    ExplicitEuler  symplectic Euler with 8 sub-steps (conditionally stable; for comparison)

An implicit step solves G(x) = shift (x - x0) + known + r(x) - f_ext = 0 for the end-of-step displacements x by
Newton's method. Each Newton update solves (shift I + K) dx = G with conjugate gradients, matrix-free: the global
matrix is never assembled, the solver only multiplies by the element tangents (ElementOperator) and uses its
diagonal as a preconditioner. The same code runs on the CPU and the GPU. Conjugate gradients need a symmetric matrix,
so the element tangents are symmetrized (all are symmetric except the linear formulation's, whose small asymmetry
only slows convergence; the residual always uses the exact forces).
"""
import torch


class ElementOperator:
    """y = K p for the sum of element matrices K (Ne, 3m, 3m) over element nodes conn (Ne, m)."""

    def __init__(self, K, conn, N):
        self.K = 0.5 * (K + K.transpose(1, 2))
        self.dof = (3 * conn[:, :, None] + torch.arange(3, device=conn.device)).reshape(len(conn), -1)
        self.N = N

    def __call__(self, p):
        y = torch.bmm(self.K, p.reshape(-1)[self.dof][:, :, None])[:, :, 0]
        return torch.zeros(3 * self.N, dtype=p.dtype, device=p.device).index_add_(0, self.dof.reshape(-1),
                                                                                  y.reshape(-1)).view(self.N, 3)

    def diagonal(self):
        d = self.K.diagonal(dim1=1, dim2=2)
        return torch.zeros(3 * self.N, dtype=d.dtype, device=d.device).index_add_(0, self.dof.reshape(-1),
                                                                                  d.reshape(-1)).view(self.N, 3)


def pcg(apply, b, inv_diag, rtol=1e-2, max_iter=500, check_every=4):
    """Jacobi-preconditioned conjugate gradients for apply(x) = b, to a relative residual `rtol`. The residual norm
    is read only every `check_every` iterations (on the GPU every read waits for the device)."""
    x = torch.zeros_like(b)
    r = b.clone()
    z = inv_diag * r
    p = z.clone()
    rz = (r * z).sum()
    stop = float((rtol * torch.linalg.norm(b)) ** 2)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = apply(p)
        alpha = rz / (p * Ap).sum()
        x += alpha * p
        r -= alpha * Ap
        if it % check_every == 0 and float((r * r).sum()) <= stop:
            break
        z = inv_diag * r
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it


class Implicit:
    """Newton's method on the end-of-step displacements; subclasses give the time-discretization coefficients."""

    def __init__(self, tol=None, max_iter=30, cg_rtol=1e-2, cg_max=500):
        self.tol, self.max_iter, self.cg_rtol, self.cg_max = tol, max_iter, cg_rtol, cg_max

    def coefficients(self, dt, v0, a0):
        """shift and the known part of the residual (per unit mass)."""
        raise NotImplementedError

    def update(self, dt, dx, v0, a0):
        """velocity and acceleration at the end of the step."""
        raise NotImplementedError

    def step(self, sim):
        model, dt, m, free = sim.model, sim.dt, sim.mass, sim.free
        x0, v0, a0 = sim.disp, sim.vel, sim.acc
        shift, known = self.coefficients(dt, v0, a0)
        shift, known = m * shift, m * known
        x = free * (x0 + dt * v0) + (1 - free) * sim.constraint
        f_ext = sim.external()
        tol = self.tol or (1e-6 if x.dtype == torch.float64 else 1e-4)
        history, cg_total, floor, stalled = [], 0, 0.0, False
        for it in range(self.max_iter + 1):
            r = model.forces(x)
            terms = (shift * (x - x0), known, r, -f_ext)
            G = free * sum(terms)
            if it == 0:   # round-off level of the residual: the size of the terms that cancel in G, times the precision
                floor = 100 * torch.finfo(x.dtype).eps * float(sum(torch.linalg.norm(free * t) for t in terms))
            g = float(torch.linalg.norm(G))
            history.append(g)
            if g <= max(tol * history[0], floor, 1e-10) or it == self.max_iter:
                break
            if x.dtype != torch.float64 and it >= 2 and g > 0.5 * history[-2]:
                stalled = True    # single precision: the residual stopped decreasing at its round-off level
                break
            if it == 0:   # the tangent at the start of the step, reused for its remaining Newton iterations
                op = ElementOperator(*model.element_matrices(), sim.N)
                inv_diag = free / (shift + op.diagonal())
            dx, k = pcg(lambda p: free * (shift * p + op(p)), G, inv_diag, self.cg_rtol, self.cg_max)
            cg_total += k
            x = x - dx
        v, a = self.update(dt, x - x0, v0, a0)
        sim.disp, sim.vel, sim.acc, sim.r_internal = x, free * v, free * a, r
        converged = history[-1] <= max(tol * history[0], floor, 1e-10) or stalled
        return dict(iterations=it, cg_iterations=cg_total, converged=bool(converged and torch.isfinite(x).all()))


class Newmark(Implicit):
    name = "Implicit Newmark"

    def coefficients(self, dt, v0, a0):
        return 4 / dt ** 2 + 2 / dt, -(4 / dt) * v0 - a0 - v0

    def update(self, dt, dx, v0, a0):
        return 2 / dt * dx - v0, 4 / dt ** 2 * (dx - dt * v0) - a0


class BackwardEuler(Implicit):
    name = "Backward Euler"

    def coefficients(self, dt, v0, a0):
        return 1 / dt ** 2 + 1 / dt, -v0 / dt

    def update(self, dt, dx, v0, a0):
        v = dx / dt
        return v, (v - v0) / dt


class ExplicitEuler:
    """v += h ((f - r(x)) / m - v); x += h v, with 8 sub-steps per frame."""

    name = "Explicit Euler (8 sub-steps)"

    def __init__(self, substeps=8):
        self.substeps = substeps

    def step(self, sim):
        h, free = sim.dt / self.substeps, sim.free
        x, v = sim.disp, sim.vel
        f_ext = sim.external()
        for _ in range(self.substeps):
            r = sim.model.forces(x)
            v = free * (v + h * ((f_ext - r) / sim.mass - v))
            x = free * (x + h * v) + (1 - free) * sim.constraint
        sim.disp, sim.vel, sim.acc, sim.r_internal = x, v, free * ((f_ext - r) / sim.mass - v), r
        return dict(iterations=self.substeps, cg_iterations=0, converged=bool(torch.isfinite(x).all()))


INTEGRATORS = {"newmark": Newmark, "backward_euler": BackwardEuler, "explicit": ExplicitEuler}
