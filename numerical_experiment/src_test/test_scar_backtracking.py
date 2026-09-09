import math

import torch

from test_ill_conditioned_inner import IllConditionedNCSC
from inner_solvers.scar import SCAR


class ShiftedQuadratic(IllConditionedNCSC):
    def __init__(self, eigenvalues, offset):
        super().__init__(eigenvalues, max_grad_calls=200_000)
        self.offset = offset

    def f(self, x, y):
        return super().f(x, y) + self.offset


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(1)
    eigenvalues = torch.logspace(-4, 0, 20)
    x = torch.randn(20)
    x /= torch.linalg.vector_norm(x)
    bar = torch.randn(20)
    solver = SCAR()

    for offset in (0.0, 1e12):
        problem = ShiftedQuadratic(eigenvalues, offset)
        for sigma in (1e-3, 0.1, 1.0):
            optimum = (x + sigma * bar) / (eigenvalues + sigma)
            y0 = torch.zeros_like(x)
            grad0 = -problem.grad_y(x, y0)
            before = problem.grad_calls
            y, count = solver._solve_ar_subproblem(
                problem, x, y0, bar, sigma, 0.1, grad_phi0=grad0,
            )
            assert count == problem.grad_calls - before
            contraction = (torch.linalg.vector_norm(y - optimum)
                           / torch.linalg.vector_norm(y0 - optimum)).item()
            gap = (0.5 * ((eigenvalues + sigma) * (y - optimum).square()).sum()).item()
            assert contraction <= 0.125, f"Distance contraction failed: {contraction}"
            before = problem.grad_calls
            M, grad, count = solver._ar_backtracking(
                problem, x, y, bar, sigma, 0.1,
            )
            assert count == problem.grad_calls - before
            assert math.isfinite(M) and M <= 2.0
            torch.testing.assert_close(grad, eigenvalues * y - x)
            d = (y - (grad + sigma * (y - bar)) / (2.0 * (M + sigma))) - y
            exact_lhs = (0.5 * (eigenvalues * d.square()).sum()).item()
            rhs = (0.5 * M * d.square().sum()).item()
            assert exact_lhs <= rhs * (1.0 + 1e-8) + 1e-30
            print(f"offset={offset:g}, sigma={sigma:g}, contraction={contraction:.6e}, "
                  f"gap={gap:.6e}, M={M:g}", flush=True)

            near_optimum = optimum + 1e-8 * torch.ones_like(x)
            before = problem.grad_calls
            M, _, count = solver._ar_backtracking(problem, x, near_optimum, bar, sigma, 0.1)
            assert count == problem.grad_calls - before
            assert math.isfinite(M) and M <= 2.0

    print("PASS: shifted objectives, exact quadratic backtracking, contraction, and oracle counts.")


if __name__ == "__main__":
    main()
