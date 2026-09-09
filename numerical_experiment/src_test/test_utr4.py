import json
import signal
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from algorithms.utr4 import UTR4
from inner_solvers.scar import SCAR
from scripts.run_comparision import evaluate
from src_test.test_adaptive_ncscproblem import CountedCosLogCosh
from subproblem_solvers.trs import TRSubproblemSolver


class QuadraticProblem:
    def __init__(self):
        self.grad_calls = 0

    def grad_y(self, x, y):
        self.grad_calls += 1
        return (x - y).detach()


class RetrySCAR(SCAR):
    def __init__(self):
        self.attempts = []

    def _ar(self, problem, x, y0, sigma1, M0, grad_phi0=None):
        self.attempts.append((y0.clone(), sigma1, M0))
        candidate = y0 if len(self.attempts) == 1 else x + 0.25 * (y0 - x)
        gradient = problem.grad_y(x, candidate)
        return candidate, M0 + 1.0, -gradient, 1


class HalfSolver(SCAR):
    def __init__(self):
        self.records = []
        self.initializations = 0

    def initialize_persistent_state(self, *args):
        self.initializations += 1
        return super().initialize_persistent_state(*args)

    def persistent_halving(self, problem, x, y, grad_y, state):
        torch.testing.assert_close(grad_y, x - y, rtol=0, atol=0)
        self.records.append(dict(x=x.clone(), y=y.clone(), state_id=id(state), M=state['M']))
        state['M'] += 1.0
        candidate = x + 0.49 * (y - x)
        gradient = problem.grad_y(x, candidate)
        return candidate, gradient, state, 1


class BranchUTR4(UTR4):
    """Controlled branch fixtures, not a numerical TR solver test."""

    def __init__(self, problem, solver, branches):
        super().__init__(problem, solver, TRSubproblemSolver(), epsilon=1e-3,
                         max_iterations=len(branches))
        self.branches = branches
        self.branch = None
        self.index = 0
        self.tuples = []
        self.values = []
        self.derivatives = []

    def _build_corrected_tuple(self, x, y, grad_y):
        self.tuples.append((x.clone(), y.clone(), grad_y.clone()))
        return 0.0, torch.ones_like(x), torch.eye(x.numel())

    def _build_corrected_value(self, x, y, grad_y):
        self.values.append(self.branch)
        return -1.0 if self.branch == 'boundary_accept' else 1.0

    def _build_corrected_derivatives(self, x, y, grad_y):
        self.derivatives.append(self.branch)
        gradient = torch.zeros_like(x) if self.branch in ('interior_pass', 'zero_pass') else torch.ones_like(x)
        return gradient, torch.eye(x.numel())

    def _solve_tr(self, g, B, radius):
        self.branch = self.branches[self.index]
        self.index += 1
        boundary = self.branch.startswith('boundary')
        length = radius if boundary else (0.0 if self.branch == 'zero_pass' else 0.1 * radius)
        d = torch.full_like(g, length)
        self._last_tr_refined = False
        return d, 0.0, length, boundary, 1e-14 * radius


def test_persistent_interface():
    p = QuadraticProblem()
    x, y = torch.zeros(1), torch.ones(1)
    gradient = p.grad_y(x, y)
    solver = RetrySCAR()
    state, count = solver.initialize_persistent_state(p, x, y, gradient)
    assert state == {'nu': 1.0, 'M': 1.0} and count == 1
    point, grad, returned, count = solver.persistent_halving(p, x, y, gradient, state)
    assert returned is state and state == {'nu': 0.25, 'M': 3.0}
    assert count == 2 and p.grad_calls == 4
    assert len(solver.attempts) == 2
    torch.testing.assert_close(solver.attempts[0][0], y, rtol=0, atol=0)
    torch.testing.assert_close(solver.attempts[1][0], y, rtol=0, atol=0)
    assert solver.attempts[1][2] == 2.0
    assert torch.linalg.vector_norm(grad).item() == 0.25
    before = p.grad_calls
    point, grad, returned, count = solver.persistent_halving(p, x, x, torch.zeros_like(x), state)
    assert count == 0 and p.grad_calls == before and len(solver.attempts) == 2
    return dict(passed=True,checks=['initial_secant_count','failed_guess_retry',
                                  'frozen_start','retained_M','nu_divided_by_four',
                                  'cached_gradient','zero_residual'])


def test_branches(branches):
    p, solver = QuadraticProblem(), HalfSolver()
    algorithm = BranchUTR4(p, solver, branches)
    result = algorithm.run(torch.zeros(1), torch.ones(1))
    h = result.history
    assert result.converged == (branches[-1] in ('interior_pass', 'zero_pass'))
    assert result.n_inner_grad_evals == p.grad_calls == sum(h['inner_grad_evals'])
    assert solver.initializations == 1
    assert len({r['state_id'] for r in solver.records}) == 1
    assert [r['M'] for r in solver.records] == list(range(1, len(solver.records) + 1))
    assert h['inner_phase'].count('working_halving') == 13 * len(branches)
    expected_rejections = sum(b in ('boundary_reject', 'interior_fail') for b in branches)
    assert h['n_rejected'] == expected_rejections
    assert h['inner_phase'].count('refresh_halving') == expected_rejections
    assert len(algorithm.tuples) == 1 + expected_rejections
    for point, _, _ in algorithm.tuples:
        torch.testing.assert_close(point, torch.zeros(1), rtol=0, atol=0)
    for i in range(1, len(h['sigma'])):
        factor = 2.0 if branches[i-1] in ('boundary_reject', 'interior_fail') else 1.0
        assert h['sigma'][i] == factor * h['sigma'][i-1]
    if result.converged:
        assert torch.linalg.vector_norm(result.x - result.y).item() == h['final_inner_residual']
        assert h['final_inner_residual'] <= h['validator_target'][-1]
        assert h['trial_status'][-1] == 'validated'
    if branches == ['boundary_reject', 'interior_pass']:
        assert 'boundary_reject' not in algorithm.derivatives
    if branches == ['zero_pass']:
        assert 'candidate_gradient' not in h['inner_phase']
        assert h['n_validators'] == 1
    return dict(branches=branches,passed=True,converged=result.converged,
                grad_calls=result.n_inner_grad_evals,phases={k:h['inner_phase'].count(k) for k in set(h['inner_phase'])})


def timeout_handler(signum, frame):
    raise TimeoutError('25-second end-to-end limit exceeded')


def run_seed2(epsilon):
    torch.manual_seed(2)
    p = CountedCosLogCosh(mu_y=1.0,omega=1.0,Q=torch.randn(5,5),beta=0.5)
    x, y = torch.randn(5), torch.zeros(5)
    algorithm = UTR4(p,SCAR(),TRSubproblemSolver(),epsilon=epsilon,max_iterations=10000)
    row = dict(seed=2,epsilon=epsilon)
    start = time.perf_counter()
    try:
        signal.alarm(25)
        result = algorithm.run(x,y)
        row.update(seconds=time.perf_counter()-start,converged=result.converged,
                   trials=result.n_iterations,grad_calls=result.n_inner_grad_evals,
                   reason=result.history['termination_reason'],
                   count_matches=result.n_inner_grad_evals==p.grad_calls==sum(result.history['inner_grad_evals']),
                   history=result.history)
        signal.alarm(25)
        M = 4 * 2 ** 0.5 * p.kappa ** 3 * p.rho
        row.update(evaluate(p,result.x,y,epsilon,M))
        row['passed'] = bool(result.converged and row['count_matches'] and row['common_test'])
    except Exception as exc:
        row.update(passed=False,error=f'{type(exc).__name__}: {exc}',actual_grad_calls=p.grad_calls)
    finally:
        signal.alarm(0)
    return row


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    signal.signal(signal.SIGALRM,timeout_handler)
    report = dict(persistent=test_persistent_interface(),branches=[],end_to_end=[])
    print(json.dumps(report['persistent']),flush=True)
    for branches in [['boundary_accept'],['boundary_reject','interior_pass'],
                     ['interior_fail','interior_pass'],['zero_pass']]:
        row = test_branches(branches)
        report['branches'].append(row)
        print(json.dumps(row),flush=True)
    for epsilon in (1e-1,1e-2):
        row = run_seed2(epsilon)
        report['end_to_end'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
    output = ROOT / 'outputs' / 'utr4_seed2_tests.json'
    output.write_text(json.dumps(report,indent=2))
    print(f'Results: {output}',flush=True)


if __name__ == '__main__':
    main()
