from .base import InnerSolverResult
from .scar import SCAR

import torch


class _EarlyStop(Exception):
    def __init__(self, y, grad_y, residual):
        self.y = y.detach().clone()
        self.grad_y = grad_y.detach().clone()
        self.residual = residual


class _CheckedProblem:
    def __init__(self, problem, solver):
        self.problem = problem
        self.solver = solver
        self.n_grad_evals = 0

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def grad_y(self, x, y):
        grad_y = self.problem.grad_y(x, y)
        self.n_grad_evals += 1
        half = self.solver.half_target
        if half is not None:
            residual = torch.linalg.vector_norm(grad_y).item()
            if residual <= self.solver.target or residual <= half:
                raise _EarlyStop(y, grad_y, residual)
        return grad_y


class SCAREarly(SCAR):
    def _ar_backtracking(self, *args, **kwargs):
        result = super()._ar_backtracking(*args, **kwargs)
        self.last_M = result[0]
        return result

    def _ar(self, problem, x, y0, sigma1, M0, grad_phi0=None):
        self.n_stages += 1
        self.half_target = 0.5 * torch.linalg.vector_norm(grad_phi0).item()
        self.last_M = M0
        before = problem.n_grad_evals
        try:
            return super()._ar(problem, x, y0, sigma1, M0, grad_phi0)
        except _EarlyStop as stop:
            if stop.residual <= self.target:
                raise
            self.n_halving_exits += 1
            return stop.y, self.last_M, -stop.grad_y, problem.n_grad_evals - before
        finally:
            self.half_target = None

    def run(
            self,
            problem,
            x,
            y0,
            *,
            stop_rule,
            target,
            max_steps=100_000,
    ) -> InnerSolverResult:
        self.target = float(target)
        self.half_target = None
        self.n_stages = 0
        self.n_halving_exits = 0
        self.n_target_exits = 0
        checked = _CheckedProblem(problem, self)
        try:
            result = super().run(
                checked, x, y0, stop_rule=stop_rule, target=target, max_steps=max_steps,
            )
        except _EarlyStop as stop:
            self.n_target_exits += 1
            return InnerSolverResult(
                y=stop.y,
                n_steps=self.n_stages,
                n_grad_evals=checked.n_grad_evals,
                grad_y=stop.grad_y,
                converged=True,
            )
        return result
