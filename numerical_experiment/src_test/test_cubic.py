"""Global cubic KKT checks, boundary geometry, and numerical regressions."""
import math
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subproblem_solvers.cubic import CubicSubproblemSolver


def solve(diagonal, gradient, M=2.):
    H = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    g = torch.tensor(gradient, dtype=torch.float64)
    r = CubicSubproblemSolver().solve(g, H, M)
    verify(g, H, M, r)
    return r


def verify(g, H, M, r):
    assert r.converged
    sn = r.s.norm().item()
    hn = torch.linalg.matrix_norm(H, ord=2).item()
    scale = g.norm().item() + (hn+r.multiplier)*sn
    assert (H@r.s+r.multiplier*r.s+g).norm() <= 2e-12*max(scale, 1e-300)
    assert torch.linalg.eigvalsh(H)[0]+r.multiplier >= -2e-12*max(hn+r.multiplier, 1e-300)
    assert abs(r.multiplier-M*sn/2) <= 2e-12*max(r.multiplier, M*sn/2, 1e-300)


def test_general_hard_case_and_repeated_minimum():
    r = solve([-2., -2., 1.], [0., 0., 3.])
    assert r.multiplier == 2. and r.n_iterations == 0
    assert abs(r.s.norm().item()-2) < 1e-14
    assert abs(r.s[2].item()+1) < 1e-14
    assert abs(r.s[:2].norm().item()-math.sqrt(3)) < 1e-14


def test_boundary_equality_and_root_with_zero_projection():
    r = solve([-2., 1.], [0., 6.])
    assert r.multiplier == 2. and r.n_iterations == 0
    r = solve([-2., 1.], [0., 9.])
    assert r.multiplier > 2. and r.n_iterations > 0


def test_zero_and_small_nonzero_gradient():
    assert solve([0., 1.], [0., 0.]).s.norm() == 0
    assert solve([-2., 1.], [0., 0.]).multiplier == 2.
    # The old ||g|| <= 1e-14 branch incorrectly returned s=0.
    r = solve([0.], [1e-20])
    assert math.isclose(r.s.item(), -1e-10, rel_tol=1e-12)


def test_near_boundary_root_below_old_cutoff():
    r = solve([-2., 1.], [1e-15, 0.])
    assert abs(r.s[0].item()+2.) < 1e-12
    assert r.n_iterations > 0  # Must not discard a gradient aligned with v_min.


def test_rotated_hard_case_and_objective_scaling():
    gen = torch.Generator().manual_seed(17)
    V, _ = torch.linalg.qr(torch.randn(4,4,generator=gen,dtype=torch.float64))
    H = V@torch.diag(torch.tensor([-2.,-2.,1.,4.],dtype=torch.float64))@V.T
    g = V@torch.tensor([0.,0.,3.,1.],dtype=torch.float64)
    for scale in (1e-12, 1., 1e12):
        r = CubicSubproblemSolver().solve(g*scale,H*scale,2*scale)
        verify(g*scale,H*scale,2*scale,r)
        assert math.isclose(r.s.norm().item(), 2., rel_tol=1e-12)


def test_random_models_and_budget_failure():
    gen = torch.Generator().manual_seed(8)
    for _ in range(30):
        H = torch.randn(5,5,generator=gen,dtype=torch.float64)
        H = (H+H.T)/2
        g = torch.randn(5,generator=gen,dtype=torch.float64)
        r = CubicSubproblemSolver().solve(g,H,1.7)
        verify(g,H,1.7,r)
        # Independent sampled model values cannot improve the certified point.
        steps=torch.randn(100,5,generator=gen,dtype=torch.float64)*3
        values=steps@g+.5*torch.einsum('bi,ij,bj->b',steps,H,steps)+1.7/6*steps.norm(dim=1)**3
        assert values.min().item() >= r.model_value-1e-12
    H=torch.eye(2,dtype=torch.float64)
    g=torch.ones(2,dtype=torch.float64)
    assert not CubicSubproblemSolver(max_iter=1).solve(g,H,2).converged


if __name__ == '__main__':
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print(test.__name__, 'OK')
