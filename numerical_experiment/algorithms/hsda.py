"""HSDA Algorithm 1, Lemma 2.3 and Theorem 2.1 (local HSDA.pdf).

Ordinary fixed-count Nesterov ascent and dense homogeneous eigenproblems.
No UTR corrected gradient, acceptance test, or persistent SCAR is used.
"""
import math
import torch
from .base import NCSCAlgorithm, AlgorithmResult


class _CountedProblem:
    def __init__(self, problem):
        self.problem = problem
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def grad_y(self, x, y):
        self.calls += 1
        return self.problem.grad_y(x, y)


class HSDA(NCSCAlgorithm):
    def __init__(self, problem, inner_solver, homogeneous_solver, epsilon,
                 max_iterations=100_000, omega=0.25, max_inner_steps=100_000):
        super().__init__(problem)
        self.inner_solver = inner_solver
        self.homogeneous_solver = homogeneous_solver
        self.ell, self.mu, self.rho = problem.ell, problem.mu, problem.rho
        self.kappa = self.ell / self.mu
        self.L1 = (1 + self.kappa) * self.ell
        self.LH = self.rho * (1 + self.kappa)**2
        self.L2 = self.rho * (1 + self.kappa)**3
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon <= min(1., self.L2 / 2):
            raise ValueError("HSDA Theorem 2.1 requires 0 < epsilon <= min(1,L2/2)")
        if not math.isfinite(omega) or not 0 < omega < .5:
            raise ValueError("omega must lie in (0,1/2)")
        for name, value in [('max_iterations', max_iterations), ('max_inner_steps', max_inner_steps)]:
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_iterations, self.max_inner_steps = int(max_iterations), int(max_inner_steps)
        self.omega = float(omega)
        self.alpha = math.sqrt(self.L2 * self.epsilon)
        self.Lambda = math.sqrt(self.epsilon / self.L2)
        self.epsilon_g = self.epsilon / 12
        self.epsilon_H = self.alpha / 12
        self.A = min(self.epsilon_g / self.ell, self.epsilon_H / (2 * self.LH))
        if not all(math.isfinite(v) and v > 0 for v in [self.alpha, self.Lambda, self.A]):
            raise ValueError("Unrepresentable HSDA parameters")

    def _inner_count(self, distance_bound):
        if distance_bound == 0:
            return 0
        log_ratio = 0.5 * math.log1p(self.kappa) + math.log(distance_bound) - math.log(self.A)
        return max(0, math.ceil(2 * math.sqrt(self.kappa) * max(0., log_ratio)))

    def run(self, x0, y0):
        x, y = x0.detach().clone(), y0.detach().clone()
        p = _CountedProblem(self.problem)
        history = {key: [] for key in ['inner_grad_evals', 'inner_phase', 'inner_steps',
                   'inner_residual', 'K_t', 'grad_norm', 'lambda_min_H', 'eigenvalue',
                   'eigen_residual', 'v', 'direction_norm', 'step_norm', 'terminal_step']}
        history.update(subproblem_type='homogeneous_eigenproblem', omega=self.omega,
                       L1=self.L1, LH=self.LH, L2=self.L2, alpha=self.alpha,
                       Lambda=self.Lambda, inner_distance_target=self.A,
                       y_matches_output_x=False)
        solves = iterations = 0

        def finish(converged, reason):
            history['termination_reason'] = reason
            history['n_oracle_builds'] = len(history['grad_norm'])
            return AlgorithmResult(x=x, y=y, n_iterations=iterations,
                                   n_inner_grad_evals=p.calls,
                                   n_subproblem_solves=solves,
                                   converged=converged, history=history)
        try:
            initial = p.grad_y(x, y)
            history['inner_grad_evals'].append(1)
            history['inner_phase'].append('initial_distance_bound')
            distance_bound = torch.linalg.vector_norm(initial).item() / self.mu
            for _ in range(self.max_iterations):
                if not math.isfinite(distance_bound):
                    raise RuntimeError('Non-finite inner distance bound')
                count = self._inner_count(distance_bound)
                if count > self.max_inner_steps:
                    return finish(False, 'Required inner steps exceed max_inner_steps')
                before = p.calls
                try:
                    inner = self.inner_solver.run(p, x, y, stop_rule='steps', target=count)
                finally:
                    history['inner_grad_evals'].append(p.calls - before)
                    history['inner_phase'].append('fixed_count_agd')
                if inner.n_grad_evals != p.calls - before or inner.n_steps != count:
                    raise RuntimeError('Inner solver count contract failed')
                if not torch.isfinite(inner.y).all():
                    raise RuntimeError('Non-finite inner solution')
                # Accuracy follows from the fixed-step distance bound.
                y = inner.y.detach()
                g = p.grad_x(x, y)
                Hxx, Hxy, Hyx, Hyy = p.hessian_blocks(x, y)
                H = Hxx - Hxy @ torch.linalg.solve(Hyy, Hyx)
                H = (H + H.T) / 2
                if not torch.isfinite(g).all() or not torch.isfinite(H).all():
                    raise RuntimeError('Non-finite raw oracle')
                history['K_t'].append(count)
                history['inner_steps'].append(inner.n_steps)
                history['inner_residual'].append(inner.residual)
                history['grad_norm'].append(g.norm().item())
                history['lambda_min_H'].append(torch.linalg.eigvalsh(H)[0].item())
                solves += 1
                sub = self.homogeneous_solver.solve(g, H, self.alpha)
                if not sub.converged:
                    raise RuntimeError('Homogeneous eigensolver failed residual check')
                u, v = sub.u, sub.v
                if abs(v) >= self.omega:
                    direction = u / v
                    # Equivalent to (2.27) for a unit eigenvector, avoiding
                    # cancellation in 1/sqrt(1+Lambda**2) near one.
                    terminal = u.norm().item() < self.Lambda * abs(v)
                else:
                    direction = u if (g @ u).item() <= 0 else -u
                    terminal = False
                norm = direction.norm().item()
                if not math.isfinite(norm) or (norm == 0 and not terminal):
                    raise RuntimeError('Invalid HSDA direction')
                step = direction if terminal else direction * (self.Lambda / norm)
                candidate = (x + step).detach()
                if not torch.isfinite(candidate).all():
                    raise RuntimeError('Non-finite HSDA step')
                for key, value in [('eigenvalue', sub.eigenvalue), ('eigen_residual', sub.residual),
                                   ('v', v), ('direction_norm', norm), ('step_norm', step.norm().item()),
                                   ('terminal_step', terminal)]:
                    history[key].append(value)
                iterations += 1
                if torch.equal(candidate, x) and not terminal:
                    return finish(False, 'Outer step stagnated at floating-point precision')
                x = candidate
                if terminal:
                    return finish(True, 'HSDA equation (2.27) satisfied; returned x+s')
                distance_bound = self.A + self.kappa * step.norm().item()
            return finish(False, 'Maximum number of HSDA iterations reached')
        except (RuntimeError, ValueError, OverflowError) as exc:
            return finish(False, str(exc))
