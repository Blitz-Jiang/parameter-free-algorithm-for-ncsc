from .scar import SCAR

import math
import torch


class _HalvingReached(Exception):
    def __init__(self, y, grad_y):
        self.y = y.detach().clone()
        self.grad_y = grad_y.detach().clone()


class _HalvingProblem:
    def __init__(self, problem, target):
        self.problem = problem
        self.target = target
        self.n_grad_evals = 0

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def grad_y(self, x, y):
        grad_y = self.problem.grad_y(x, y)
        self.n_grad_evals += 1
        if torch.linalg.vector_norm(grad_y).item() <= self.target:
            raise _HalvingReached(y, grad_y)
        return grad_y


class SCARPersistentEarly(SCAR):
    def _ar_backtracking(self, *args, **kwargs):
        result = super()._ar_backtracking(*args, **kwargs)
        self.last_M = result[0]
        return result

    def _ar(self, problem, x, y0, sigma1, M0, grad_phi0=None):
        initial_evals = 0
        if grad_phi0 is None:
            grad_phi0 = -problem.grad_y(x, y0)
            initial_evals = 1
        residual = torch.linalg.vector_norm(grad_phi0).item()
        if not math.isfinite(residual):
            raise RuntimeError("Non-finite initial gradient in persistent early AR.")
        self.last_M = float(M0)
        if residual == 0.0:
            return y0.detach(), self.last_M, grad_phi0.detach(), initial_evals

        checked = _HalvingProblem(problem, 0.5 * residual)
        try:
            y, M, grad_phi, n_grad_evals = super()._ar(
                checked, x, y0, sigma1, M0, grad_phi0,
            )
        except _HalvingReached as stop:
            return (
                stop.y, self.last_M, -stop.grad_y,
                initial_evals + checked.n_grad_evals,
            )
        if n_grad_evals != checked.n_grad_evals:
            raise RuntimeError("Persistent early AR gradient count mismatch.")
        return y, M, grad_phi, initial_evals + n_grad_evals
