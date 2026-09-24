import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.utr5 import UTR5
from inner_solvers.scar import SCAR
from problems.CosLogCosh import CosLogCosh
from subproblem_solvers.trs import TRSubproblemSolver


def test_fixed_count_and_default_regression():
    torch.manual_seed(2)
    Q, x, y = torch.randn(5,5), torch.randn(5), torch.zeros(5)
    rows = {}
    for mode in ['target','fixed_count']:
        p = CosLogCosh(mu_y=.1,omega=1.,Q=Q,beta=.5)
        a = UTR5(p,SCAR(max_halving_attempts=4),TRSubproblemSolver(),.001,tracking_mode=mode)
        r = a.run(x,y); h = r.history
        assert r.converged, h['termination_reason']
        assert r.n_inner_grad_evals == sum(h['inner_grad_evals'])
        actual = sum(phase == 'working_halving' for phase in h['inner_phase'])
        required = sum(h['tracking_halvings'])
        assert actual == required if mode == 'fixed_count' else actual < required
        rows[mode] = r
    assert rows['target'].n_inner_grad_evals == 4872
    assert rows['fixed_count'].n_inner_grad_evals == 8059
    torch.testing.assert_close(rows['target'].x, rows['fixed_count'].x,atol=1e-8,rtol=1e-8)


def test_failed_halving_is_not_success():
    class NoProgress(SCAR):
        def _ar(self, **kw):
            self.attempts += 1
            return kw['y0'],kw['M0'],kw['grad_phi0'],0
    solver = NoProgress(max_halving_attempts=3); solver.attempts=0
    state = {'nu':1.,'M':1.}
    try:
        solver.persistent_halving(None,torch.zeros(1),torch.ones(1),torch.ones(1),state)
    except RuntimeError as error:
        assert 'No successful halving is claimed' in str(error)
        assert solver.attempts == 3 and state['nu'] == 1/64
    else:
        raise AssertionError('Failure was silently accepted')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    test_fixed_count_and_default_regression(); print('PASS fixed-count and default regression')
    test_failed_halving_is_not_success(); print('PASS explicit halving budget failure')
