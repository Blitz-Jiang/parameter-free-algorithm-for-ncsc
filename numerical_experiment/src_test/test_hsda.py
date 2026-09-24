"""Math-property and branch tests; run directly, no pytest dependency."""
import sys
from pathlib import Path
import math
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.hsda import HSDA
from inner_solvers.nesterov import NesterovAGD
from subproblem_solvers.homogeneous import HomogeneousSubproblemSolver
from problems.NCSC import NCSCProblem


class Problem(NCSCProblem):
    def __init__(self, saddle=False):
        super().__init__()
        self.saddle = saddle
        self.calls = 0

    def _calculate_constants(self):
        return dict(ell=2., mu=1., rho=1.)

    def f(self, x, y):
        primal = torch.cos(x).sum() if self.saddle else .5 * x.square().sum()
        return primal + .25 * (x @ y) - .5 * y.square().sum()

    def grad_y(self, x, y):
        self.calls += 1
        return .25 * x - y


def make(p, eps=.1, **kwargs):
    return HSDA(p, NesterovAGD(p.ell, p.mu), HomogeneousSubproblemSolver(), eps, **kwargs)


def test_eigenproblem():
    solver = HomogeneousSubproblemSolver()
    for g, H in [(torch.tensor([.3, -.2]), torch.diag(torch.tensor([-2., 1.]))),
                 (torch.zeros(2), torch.diag(torch.tensor([-2., 1.]))),
                 (torch.zeros(2), torch.eye(2))]:
        r = solver.solve(g, H, .1)
        G = torch.cat([torch.cat([H, g[:, None]], 1),
                       torch.cat([g, torch.tensor([-.1])])[None, :]], 0)
        w = torch.cat([r.u, torch.tensor([r.v])])
        torch.testing.assert_close(w.norm(), torch.tensor(1.))
        torch.testing.assert_close(G @ w, r.eigenvalue * w)
        assert r.converged
        # The returned eigenvector attains a lower quadratic value than probes.
        probes = torch.randn(3, 100); probes /= probes.norm(dim=0)
        assert torch.all((probes * (G @ probes)).sum(0) >= r.eigenvalue - 1e-12)


def test_terminal_full_step_and_budget():
    p = Problem(); a = make(p)
    x = torch.tensor([.02]); r = a.run(x, torch.tensor([.7]))
    assert r.converged and r.n_iterations == 1
    assert r.history['terminal_step'] == [True]
    step = (r.x - x).norm().item()
    assert math.isclose(step, r.history['direction_norm'][0], rel_tol=1e-12)
    assert step < a.Lambda
    assert (r.y - .25 * x).norm() <= a.A
    assert r.n_inner_grad_evals == p.calls == sum(r.history['inner_grad_evals'])
    assert r.n_inner_grad_evals == 1 + sum(r.history['K_t'])
    assert (1.0625 * r.x).norm() < (1.0625 * x).norm()


def test_saddle_nonzero_step():
    p = Problem(saddle=True); a = make(p, eps=.01, max_iterations=1)
    r = a.run(torch.zeros(1), torch.zeros(1))
    assert not r.converged
    assert abs(r.history['v'][0]) < a.omega
    assert not r.history['terminal_step'][0]
    assert math.isclose(r.x.norm().item(), a.Lambda, rel_tol=1e-12)
    def P(x):
        return torch.cos(x).sum() + .03125 * x.square().sum()
    assert P(r.x) < P(torch.zeros(1))


def test_fixed_steps_accuracy_and_warm_start():
    p = Problem(); a = make(p, eps=.01)
    x = torch.tensor([.2]); y = torch.tensor([1.])
    bound = p.grad_y(x, y).norm().item() / p.mu
    first = a.inner_solver.run(p, x, y, stop_rule='steps', target=a._inner_count(bound))
    assert (first.y - .25*x).norm() <= a.A
    xp = x + .07
    second = a.inner_solver.run(p, xp, first.y, stop_rule='steps',
                               target=a._inner_count(a.A + a.kappa * (xp-x).norm().item()))
    assert (second.y - .25*xp).norm() <= a.A
    limited = make(Problem(), max_inner_steps=1).run(x, y)
    assert not limited.converged and limited.n_subproblem_solves == 0


def test_convergence_and_zero_stationary():
    p = Problem(); a = make(p, eps=.01, max_iterations=200)
    r = a.run(torch.tensor([.2]), torch.tensor([1.]))
    assert r.converged and r.n_iterations > 1
    assert (1.0625*r.x).norm() < .01
    assert all(math.isclose(s, a.Lambda, rel_tol=1e-12)
               for s in r.history['step_norm'][:-1])
    p = Problem(); r = make(p).run(torch.zeros(1), torch.zeros(1))
    assert r.converged and r.x.norm() == 0


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1); torch.manual_seed(1)
    for test in [test_eigenproblem, test_terminal_full_step_and_budget,
                 test_saddle_nonzero_step, test_fixed_steps_accuracy_and_warm_start,
                 test_convergence_and_zero_stationary]:
        test(); print(f'PASS {test.__name__}')
