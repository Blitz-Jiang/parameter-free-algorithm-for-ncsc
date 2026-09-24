"""Reproduce fixed-count float64 SCAR failure and replay its first inner task
with 80-digit ordinary SCAR. High precision is diagnostic, not a new benchmark.
"""
import json
import math
import sys
from pathlib import Path
import torch
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.utr5 import UTR5
from inner_solvers.scar import SCAR
from problems.CosLogCosh import CosLogCosh
from subproblem_solvers.trs import TRSubproblemSolver


class TracedSCAR(SCAR):
    def __init__(self):
        super().__init__(max_halving_attempts=4)
        self.calls = []

    def persistent_halving(self, problem, x, y, grad_y, state):
        entry = dict(x=x.tolist(), y=y.tolist(), residual=grad_y.norm().item(),
                     nu=state['nu'], M=state['M'], attempts=[])
        self.calls.append(entry)
        return super().persistent_halving(problem, x, y, grad_y, state)

    def _ar(self, *args, **kwargs):
        result = super()._ar(*args, **kwargs)
        y, M, grad, count = result
        self.calls[-1]['attempts'].append(dict(residual=grad.norm().item(), M=M,
                                               sigma1=kwargs['sigma1'], gradients=count))
        return result


def high_precision_replay(problem, entry, count):
    """Scalar-list port of SCAR._ar/_solve_ar_subproblem/_ar_backtracking.

    Same fixed x, nu/M and warm y; analytical inner gradient at higher precision.
    Uses neither Newton nor direct inner minimization in the SCAR replay.
    """
    mp.mp.dps = 80
    x, y = [mp.mpf(v) for v in entry['x']], [mp.mpf(v) for v in entry['y']]
    mu, beta, a = map(mp.mpf, [problem.mu_y, problem.beta, problem.a_mu])
    Q = [[mp.mpf(v) for v in row] for row in problem.Q.tolist()]
    b = [a*sum(x[i]*Q[i][j] for i in range(len(x))) for j in range(len(y))]
    nu, M = mp.mpf(entry['nu']), mp.mpf(entry['M'])
    evals = 0
    def norm(v): return mp.sqrt(sum(t*t for t in v))
    def grad(v):
        nonlocal evals
        evals += 1
        return [mu*t+beta*mp.tanh(t)-bj for t,bj in zip(v,b)]
    def value(v): return sum(mu*t*t/2+beta*mp.log(mp.cosh(t))-bj*t for t,bj in zip(v,b))
    def line_ok(v, vn, gv, L):
        d = [j-i for i,j in zip(v,vn)]
        # Stable value remainder, matching CosLogCosh.value_remainder_y.
        # Direct subtraction can also lose all digits with arbitrary precision.
        remainder = mp.mpf(0)
        for yi,di in zip(v,d):
            ai = -2*di if yi >= 0 else 2*di
            qi = 1/(1+mp.exp(2*abs(yi)))
            w = qi*mp.expm1(ai)
            if abs(ai) < mp.mpf('1e-8'):
                er = sum(ai**k/mp.factorial(k) for k in range(2,22))
                lr = sum((-1)**(k+1)*w**k/k for k in range(2,22))
                rem = qi*er+lr
            else:
                rem = mp.log1p(w)-qi*ai
            remainder += mu*di*di/2+beta*rem
        return remainder <= L*sum(t*t for t in d)/2
    def subsolve(y0, bar, sigma, L0, g0):
        y, v, theta, L = y0[:], y0[:], mp.mpf(1), L0
        for k in range(1, 100001):
            gv = g0 if k == 1 and g0 is not None else grad(v)
            for _ in range(100):
                yn = [vi+(-gi+sigma*(bi-vi))/(L+sigma) for vi,gi,bi in zip(v,gv,bar)]
                if line_ok(v,yn,gv,L): break
                L *= 2
            else: raise RuntimeError('MP backtracking limit')
            if k >= int(mp.ceil(8*mp.sqrt(4*L/sigma))): return yn
            tn = (1+mp.sqrt(1+4*theta*theta))/2
            v = [j+(theta-1)/tn*(j-i) for i,j in zip(y,yn)]
            y, theta = yn, tn
        raise RuntimeError('MP subproblem limit')
    def ar(u, gu, nu, M):
        yp, bar, sigma_prev, gp = u[:], u[:], mp.mpf(0), gu
        sigma = nu/10
        for _ in range(100):
            gamma = 1-sigma_prev/sigma
            bar = [(1-gamma)*bi+gamma*yi for bi,yi in zip(bar,yp)]
            yn = subsolve(yp,bar,sigma,M/2,gp)
            gn = grad(yn); gs = [gi+sigma*(yi-bi) for gi,yi,bi in zip(gn,yn,bar)]
            M /= 2
            for _ in range(100):
                trial = [yi-gi/(2*(M+sigma)) for yi,gi in zip(yn,gs)]
                if line_ok(yn,trial,gn,M): break
                M *= 2
            else: raise RuntimeError('MP AR backtracking limit')
            if sigma >= M: return yn, M, gn
            yp, gp, sigma_prev = yn, gn, sigma
            sigma *= 4
        raise RuntimeError('MP AR stage limit')
    g = grad(y)
    residuals, failures = [str(norm(g))], 0
    for _ in range(count):
        R = norm(g)
        for _ in range(30):
            yn, M, gn = ar(y,g,nu,M)
            if norm(gn) <= R/2:
                y,g = yn,gn
                break
            failures += 1
            nu /= 4
        else: raise RuntimeError('MP halving limit')
        residuals.append(str(norm(g)))
    rounded = [mp.mpf(float(t)) for t in y]
    return dict(dps=mp.mp.dps, successful_halvings=count, failures=failures,
                gradient_evaluations=evals, residuals=residuals,
                residual_after_rounding_y_to_float64=str(norm(grad(rounded))))


def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1); torch.manual_seed(2)
    Q, x, y = torch.randn(5,5), torch.randn(5), torch.zeros(5)
    p = CosLogCosh(mu_y=1., omega=1., Q=Q, beta=.5)
    rows = {}
    for mode in ['target','fixed_count']:
        inner = TracedSCAR()
        result = UTR5(p,inner,TRSubproblemSolver(),.001, tracking_mode=mode).run(x,y)
        rows[mode] = dict(converged=result.converged, Q=result.n_subproblem_solves,
                          Ny=result.n_inner_grad_evals, history=result.history, calls=inner.calls)
        print(mode, result.converged, result.history['termination_reason'], flush=True)
    strict = rows['fixed_count']
    initial_halvings = sum(phase == 'initial_refinement' for phase in strict['history']['inner_phase'])
    entry = strict['calls'][initial_halvings]
    N = strict['history']['tracking_halvings'][0]
    rows['high_precision_first_candidate'] = high_precision_replay(p, entry, N)
    residuals = [mp.mpf(r) for r in rows['high_precision_first_candidate']['residuals']]
    assert all(b <= a/2 for a,b in zip(residuals,residuals[1:]))
    assert len(residuals) == N+1
    out = Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/utr5_fixed_diagnosis.json')
    out.write_text(json.dumps(rows,indent=2))
    hp = rows['high_precision_first_candidate']
    print('MP replay:', N, 'halvings;', hp['failures'], 'failures; residual', hp['residuals'][-1])
    print('After float64 rounding:', hp['residual_after_rounding_y_to_float64'])
    print('Output:',out)


if __name__ == '__main__':
    main()
