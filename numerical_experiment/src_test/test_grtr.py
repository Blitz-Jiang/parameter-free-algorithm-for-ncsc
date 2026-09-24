"""GRTR mathematical properties and integration contracts."""
import sys
from pathlib import Path
import math
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.grtr import GRTR
from inner_solvers.nesterov import NesterovAGD
from subproblem_solvers.trs import TRSubproblemSolver, TRSubproblemResult
from src_test.test_hsda import Problem


def make(p, epsilon=.001, **kwargs):
    return GRTR(p, NesterovAGD(p.ell, p.mu), TRSubproblemSolver(), epsilon, **kwargs)


def test_terminal_returns_current_point():
    p = Problem(); a = make(p)
    x = torch.tensor([.0001]); r = a.run(x, torch.tensor([.7]))
    assert r.converged and r.n_iterations == 1
    torch.testing.assert_close(r.x, x, rtol=0, atol=0)
    assert r.history['step_norm'][0] > 0  # Computed step must NOT be taken.
    assert r.history['n_updates'] == 0 and r.history['y_matches_output_x']
    assert r.n_inner_grad_evals == p.calls == sum(r.history['inner_grad_evals'])
    assert r.n_inner_grad_evals == 1 + sum(r.history['K_t'])
    assert (r.y-.25*x).norm() <= a.A
    gnorm = r.history['grad_norm'][0]
    assert math.isclose(r.history['regularization'][0], a.sigma*math.sqrt(gnorm))
    assert math.isclose(r.history['radius'][0], a.radius_scale*math.sqrt(a.epsilon))
    assert (1.0625*r.x).norm() <= (97/96)*a.epsilon


def test_negative_curvature_and_small_gradient():
    p = Problem(saddle=True); a = make(p, max_iterations=1)
    x = torch.zeros(1); r = a.run(x, x)
    assert not r.converged
    assert r.history['regularization'][0] == 0
    assert r.history['multiplier'][0] > math.sqrt(a.L2*a.epsilon)
    assert r.history['n_updates'] == 1
    assert math.isclose(r.x.norm().item(), a.radius_scale*math.sqrt(a.epsilon))
    assert torch.cos(r.x).sum()+.03125*r.x.square().sum() < 1


def test_convergence_and_error_budgets():
    p = Problem(); a = make(p, max_iterations=2000)
    r = a.run(torch.tensor([.3]), torch.tensor([1.]))
    assert r.converged and r.n_iterations > 1
    assert r.history['n_updates'] == r.n_iterations-1
    assert (1.0625*r.x).norm() <= (97/96)*a.epsilon
    assert (r.y-.25*r.x).norm() <= a.A
    assert max(max(m.values()) for m in r.history['tr_kkt']) <= a.kkt_tol
    assert math.isclose(a.epsilon_g, min(1/96, math.sqrt(a.L2)/(16*a.L1))*a.epsilon**1.5)
    # Known y*: verify fixed-count initialization and warm-start distances.
    x, y = torch.tensor([.4]), torch.tensor([1.])
    inner = a.inner_solver.run(p, x, y, stop_rule='steps',
                              target=a._inner_count(p.grad_y(x,y).norm().item()/p.mu))
    assert (inner.y-.25*x).norm() <= a.A
    xp = x+.05
    nxt = a.inner_solver.run(p, xp, inner.y, stop_rule='steps',
                            target=a._inner_count(a.A+a.kappa*.05))
    assert (nxt.y-.25*xp).norm() <= a.A


def test_failed_subproblem_and_budget():
    class Broken:
        def solve(self, g, H, radius):
            return TRSubproblemResult(torch.zeros_like(g), 0., 0., 1, True)
    p = Problem(); a = make(p); a.trust_region_solver = Broken()
    r = a.run(torch.tensor([.1]), torch.zeros(1))
    assert not r.converged and 'KKT' in r.history['termination_reason']
    torch.testing.assert_close(r.x, torch.tensor([.1]))
    p = Problem(); r = make(p, max_inner_steps=1).run(torch.ones(1), torch.ones(1))
    assert not r.converged and r.n_subproblem_solves == 0


def test_stationary_zero_and_kappa_one():
    p = Problem(); r = make(p).run(torch.zeros(1), torch.zeros(1))
    assert r.converged and r.history['step_norm'] == [0.]
    class UnitProblem(Problem):
        def _calculate_constants(self):
            return dict(ell=1., mu=1., rho=1.)
        def f(self,x,y):
            return .5*x.square().sum()-.5*y.square().sum()
        def grad_y(self,x,y):
            self.calls += 1
            return -y
    p = UnitProblem(); r = make(p).run(torch.zeros(1), torch.ones(1))
    assert r.converged and r.y.norm() == 0


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    for test in [test_terminal_returns_current_point, test_negative_curvature_and_small_gradient,
                 test_convergence_and_error_budgets, test_failed_subproblem_and_budget,
                 test_stationary_zero_and_kappa_one]:
        test(); print(f'PASS {test.__name__}')
