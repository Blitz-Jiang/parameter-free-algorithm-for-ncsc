import sys
from pathlib import Path
import torch
import mpmath
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from problems.CosLogCosh import CosLogCosh
from problems.coslogcosh_mp import CosLogCoshMP
from inner_solvers.scar_mp import SCARMP
from subproblem_solvers.trs_mp import TRSubproblemSolverMP
from algorithms.utr5_fixed_mp import UTR5FixedMP


def context():
    c=mpmath.mp.clone(); c.dps=80
    return c


def test_analytic_oracle_and_remainder():
    c=context()
    p=CosLogCosh(mu_y=.1,omega=1.3,Q=torch.tensor([[.3,-.2],[.5,.7]]),beta=.5)
    q=CosLogCoshMP(p,c)
    x,y=c.matrix([.2,-.4]),c.matrix([.3,-.7])
    gy=q.grad_y(x,y)
    for j in range(2):
        def f(t):
            z=y.copy();z[j]=t
            return q.f(x,z)
        assert abs(c.diff(f,y[j])-gy[j])<c.mpf('1e-70')
    tx,ty=torch.tensor([float(t) for t in x]),torch.tensor([float(t) for t in y])
    val,g,H=q.corrected(x,y,gy)
    xx,xy,yx,yy=p.hessian_blocks(tx,ty)
    tg=p.grad_x(tx,ty)-xy@torch.linalg.solve(yy,p.grad_y(tx,ty))
    tH=xx-xy@torch.linalg.solve(yy,yx)
    torch.testing.assert_close(torch.tensor([float(t) for t in g]),tg,atol=1e-13,rtol=1e-13)
    torch.testing.assert_close(torch.tensor([[float(t) for t in row] for row in H.tolist()]),tH,atol=1e-13,rtol=1e-13)
    d=c.matrix([c.mpf('1e-45'),-c.mpf('2e-45')])
    remainder=q.q_remainder(y,y+d)
    quadratic=sum((q.mu+q.beta*(1-c.tanh(v)**2))*di**2/2 for v,di in zip(y,d))
    assert abs(remainder/quadratic-1)<c.mpf('1e-34')


def test_mp_tr_cases():
    c=context(); solver=TRSubproblemSolverMP(c)
    for g,H,R in [(c.matrix([1,0]),c.diag([2,3]),c.mpf(2)),
                  (c.matrix([0,0]),c.diag([-2,1]),c.mpf(1)),
                  (c.matrix([1,2]),c.diag([-2,1]),c.mpf('.3')),
                  (c.matrix([0,0]),c.diag([0,1]),c.mpf(1))]:
        s,lam,r,boundary=solver.solve(g,H,R)
        assert c.norm((H+lam*c.eye(2))*s+g)<c.mpf('1e-35')
        assert r <= R*(1+c.mpf('1e-35'))
        assert c.eigsy(H+lam*c.eye(2),eigvals_only=True)[0]>=-c.mpf('1e-35')


def test_scar_preserves_sub_float64_residual():
    c=context()
    q=CosLogCoshMP(CosLogCosh(mu_y=1.,omega=1.,Q=torch.tensor([[1.]]),beta=.5),c)
    solver=SCARMP(c)
    x,y=c.matrix([.7]),c.matrix([0])
    g=q.grad_y(x,y);a=solver.secant(q,x,y,g);state=dict(nu=a,M=a)
    for _ in range(22):
        R=c.norm(g); y,g=solver.halve(q,x,y,g,state)
        assert c.norm(g)<=R/2
    assert c.norm(g)<c.mpf('1e-20')
    assert c.norm(q.grad_y(x,c.matrix([float(t) for t in y])))>c.norm(g)*100


def test_full_run_and_count_independence():
    class Spy(CosLogCoshMP):
        def __init__(self,p,c):
            super().__init__(p,c);self.independent_calls=0
        def grad_y(self,*args):
            self.independent_calls+=1
            return super().grad_y(*args)
    torch.manual_seed(2)
    p=CosLogCosh(mu_y=1.,omega=1.,Q=torch.randn(2,2),beta=.5)
    old_dps=mpmath.mp.dps
    a=UTR5FixedMP(p,lambda c:Spy(p,c),.001,dps=80,max_iterations=100)
    r=a.run(torch.randn(2),torch.zeros(2));h=r.history
    assert r.converged,h['termination_reason']
    assert h['tracking_halvings']==h['working_halvings_completed']
    assert r.n_inner_grad_evals==a.oracle.independent_calls==sum(h['inner_grad_evals'])
    assert mpmath.mp.dps==old_dps
    assert h['decimal_digits']==80 and h['arithmetic']=='mpmath'


if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    for test in [test_analytic_oracle_and_remainder,test_mp_tr_cases,
                 test_scar_preserves_sub_float64_residual,test_full_run_and_count_independence]:
        test();print('PASS',test.__name__,flush=True)
