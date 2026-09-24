"""Observable initialization and query-free fixed-step AGD endpoints."""
import sys, math
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_test.test_hsda import Problem
from inner_solvers.nesterov import NesterovAGD
from algorithms.mcn import MCN
from algorithms.grtr import GRTR
from algorithms.hsda import HSDA
from subproblem_solvers.cubic import CubicSubproblemSolver
from subproblem_solvers.trs import TRSubproblemSolver
from subproblem_solvers.homogeneous import HomogeneousSubproblemSolver


def test_agd_trajectory_and_queries():
    p=Problem(); solver=NesterovAGD(p.ell,p.mu)
    x=torch.tensor([.4]); y=torch.tensor([1.])
    for n in (0,1,7):
        before=p.calls
        out=solver.run(p,x,y,stop_rule='steps',target=n)
        assert p.calls-before==out.n_grad_evals==n
        assert out.grad_y is None and out.residual is None
        expected=y.clone(); v=y.clone()
        for _ in range(n):
            new=v+solver.step_size*p.grad_y(x,v)
            v=new+solver.beta*(new-expected); expected=new
        torch.testing.assert_close(out.y,expected,rtol=0,atol=0)


def test_mcn_initialization_and_repeated_runs():
    p=Problem(); solver=NesterovAGD(p.ell,p.mu)
    algo=MCN(p,solver,CubicSubproblemSolver(),epsilon=.001,max_iterations=1000)
    x=torch.tensor([.3])
    for y in (torch.tensor([1.]),torch.tensor([-.8])):
        D=p.grad_y(x,y).norm().item()/p.mu
        expected=max(0,math.ceil(2*math.sqrt(p.kappa)*math.log(math.sqrt(p.kappa+1)*D/algo.inner_distance_tol)))
        before=p.calls
        out=algo.run(x,y)
        assert out.converged and out.history['K0']==expected
        assert out.n_inner_grad_evals==p.calls-before==1+sum(out.history['K_t'])
        assert out.n_inner_grad_evals==sum(out.history['inner_grad_evals'])
        assert all(r is None for r in out.history['inner_residual'])


def test_counts_allow_zero_and_use_exact_prefactor():
    p=Problem(); inner=NesterovAGD(p.ell,p.mu)
    algos=[GRTR(p,inner,TRSubproblemSolver(),.001),
           HSDA(p,inner,HomogeneousSubproblemSolver(),.001)]
    for a in algos:
        assert a._inner_count(0)==0
        assert a._inner_count(a.A/(2*math.sqrt(a.kappa+1)))==0
        for D in (a.A, 10*a.A, .7):
            expected=max(0,math.ceil(2*math.sqrt(a.kappa)*math.log(math.sqrt(a.kappa+1)*D/a.A)))
            assert a._inner_count(D)==expected


if __name__=='__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS',name)
