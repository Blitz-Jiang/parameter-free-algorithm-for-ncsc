import argparse
import json
import signal
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from algorithms.utr3 import UTR3
from inner_solvers.scar import SCAR
from inner_solvers.scar_early import SCAREarly
from scripts.run_comparision import evaluate
from src_test.test_adaptive_ncscproblem import CountedCosLogCosh
from subproblem_solvers.trs import TRSubproblemSolver


class LoggedSolver:
    def __init__(self, solver, problem):
        self.solver = solver
        self.problem = problem
        self.calls = []

    def run(self, problem, x, y0, **kwargs):
        before = self.problem.grad_calls
        start = time.perf_counter()
        result = self.solver.run(problem, x, y0, **kwargs)
        actual = self.problem.grad_calls - before
        assert actual == result.n_grad_evals
        assert result.converged and result.residual <= kwargs['target']
        self.calls.append(dict(
            call=len(self.calls), grad_calls=actual, stages=result.n_steps,
            residual=result.residual, seconds=time.perf_counter() - start,
            halving_exits=getattr(self.solver, 'n_halving_exits', 0),
            target_exits=getattr(self.solver, 'n_target_exits', 0),
        ))
        return result


def timeout_handler(signum, frame):
    raise TimeoutError('Per-algorithm time limit exceeded.')


def main():
    parser = argparse.ArgumentParser(description='UTR3: current SCAR vs residual early stopping')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2])
    parser.add_argument('--epsilon', type=float, default=1e-7)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--output', type=Path, default=Path('/tmp/scar_early_stopping.json'))
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    signal.signal(signal.SIGALRM, timeout_handler)
    report = dict(settings=dict(seeds=args.seeds, epsilon=args.epsilon,
                               d=5, mu_y=1.0, omega=1.0, beta=0.5), runs=[])
    for seed in args.seeds:
        torch.manual_seed(seed)
        Q, x0, y0 = torch.randn(5, 5), torch.randn(5), torch.zeros(5)
        for name, solver in [('current', SCAR()), ('early_stopping', SCAREarly())]:
            problem = CountedCosLogCosh(mu_y=1.0, omega=1.0, Q=Q.clone(), beta=0.5)
            logged = LoggedSolver(solver, problem)
            algorithm = UTR3(
                problem, logged, TRSubproblemSolver(tol=1e-14, max_iter=1000),
                epsilon=args.epsilon, max_iterations=1000,
            )
            row = dict(seed=seed, method=name)
            start = time.perf_counter()
            try:
                signal.alarm(args.timeout)
                result = algorithm.run(x0.clone(), y0.clone())
                elapsed = time.perf_counter() - start
                history = result.history
                assert result.n_inner_grad_evals == problem.grad_calls
                assert result.n_inner_grad_evals == sum(c['grad_calls'] for c in logged.calls)
                assert result.n_inner_grad_evals == sum(history['inner_grad_evals'])
                for i, call in enumerate(logged.calls):
                    call['status'] = 'initial' if i == 0 else history['trial_status'][i - 1]
                row.update(
                    converged=result.converged, seconds=elapsed,
                    reason=history['termination_reason'], trials=result.n_iterations,
                    accepted=history['n_accepted'], rejected=history['n_rejected'],
                    grad_calls=result.n_inner_grad_evals, scar_calls=len(logged.calls),
                    stages=sum(c['stages'] for c in logged.calls), count_matches=True,
                    halving_exits=sum(c['halving_exits'] for c in logged.calls),
                    target_exits=sum(c['target_exits'] for c in logged.calls),
                    final_x=result.x.tolist(), calls=logged.calls,
                )
                signal.alarm(args.timeout)
                M = 4.0 * 2.0 ** 0.5 * problem.kappa ** 3 * problem.rho
                row.update(evaluate(problem, result.x, y0, args.epsilon, M))
                row['passed'] = bool(result.converged and row['common_test'])
            except Exception as exc:
                row.update(passed=False, error=f'{type(exc).__name__}: {exc}')
                row.setdefault('seconds', time.perf_counter() - start)
                row.setdefault('calls', logged.calls)
            finally:
                signal.alarm(0)
            report['runs'].append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ('calls', 'final_x')}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f'Results: {args.output}', flush=True)
    if not all(row['passed'] for row in report['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
