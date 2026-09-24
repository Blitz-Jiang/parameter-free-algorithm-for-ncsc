"""GRTR Algorithm 1: fixed-count AGD and Lemma 2.5 parameters.

Stopping uses Lemma 2.6 at x_t, after solving its TR model.
"""
import math
import torch
from .base import NCSCAlgorithm, AlgorithmResult


class _CountedProblem:
    def __init__(self, problem):
        self.problem, self.calls = problem, 0

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def grad_y(self, x, y):
        self.calls += 1
        return self.problem.grad_y(x, y)


class GRTR(NCSCAlgorithm):
    def __init__(self, problem, inner_solver, trust_region_solver, epsilon,
                 max_iterations=100_000, max_inner_steps=100_000, kkt_tol=1e-8):
        super().__init__(problem)
        self.inner_solver, self.trust_region_solver = inner_solver, trust_region_solver
        self.ell, self.mu, self.rho = problem.ell, problem.mu, problem.rho
        self.kappa = self.ell / self.mu
        self.L1 = (1 + self.kappa) * self.ell
        self.LH = self.rho * (1 + self.kappa)**2
        self.L2 = self.rho * (1 + self.kappa)**3
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon <= 1:
            raise ValueError('GRTR implementation requires 0 < epsilon <= 1')
        for name, value in [('max_iterations', max_iterations), ('max_inner_steps', max_inner_steps)]:
            if int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(kkt_tol) or kkt_tol <= 0:
            raise ValueError('kkt_tol must be positive and finite')
        self.max_iterations, self.max_inner_steps = int(max_iterations), int(max_inner_steps)
        self.kkt_tol = float(kkt_tol)
        # Equation (2.11): epsilon_1 is a constant times epsilon^(3/2).
        self.sigma = math.sqrt(self.L2) / 2
        self.radius_scale = 1 / (4 * math.sqrt(self.L2))
        self.epsilon_g = min(1/96, math.sqrt(self.L2)/(16*self.L1)) * self.epsilon**1.5
        self.epsilon_H = math.sqrt(self.L2 * self.epsilon) / 12
        self.A = min(self.epsilon_g/self.ell, self.epsilon_H/(2*self.LH))
        if not all(math.isfinite(v) and v > 0 for v in
                   [self.sigma, self.radius_scale, self.epsilon_g, self.epsilon_H, self.A]):
            raise ValueError('Unrepresentable GRTR parameters')

    def _inner_count(self, distance_bound):
        if distance_bound == 0:
            return 0
        log_ratio = 0.5 * math.log1p(self.kappa) + math.log(distance_bound) - math.log(self.A)
        return max(0, math.ceil(2*math.sqrt(self.kappa)*max(0., log_ratio)))

    def _check_tr(self, g, B, radius, result):
        s, lam = result.s, float(result.multiplier)
        if not result.converged or not torch.isfinite(s).all() or not math.isfinite(lam):
            raise RuntimeError('TR solver failed or returned non-finite data')
        norm = s.norm().item()
        bnorm = torch.linalg.matrix_norm(B, ord=2).item()
        stationarity = (B@s + lam*s + g).norm().item()
        eigen_min = torch.linalg.eigvalsh(B)[0].item() + lam
        tol = max(self.kkt_tol, 64*torch.finfo(g.dtype).eps)
        # Relative radius accuracy is essential for small trust regions.
        metrics = dict(stationarity=stationarity/max(g.norm().item()+(bnorm+abs(lam))*norm, 1e-30),
                       feasibility=max(0., norm-radius)/radius,
                       complementarity=abs(lam*(norm-radius))/max(abs(lam)*radius, 1e-30),
                       psd=max(0., -eigen_min)/max(1., bnorm+abs(lam)),
                       dual=max(0., -lam)/max(1., abs(lam)))
        if max(metrics.values()) > tol:
            raise RuntimeError(f'TR KKT check failed: {metrics}')
        return metrics

    def run(self, x0, y0):
        x, y = x0.detach().clone(), y0.detach().clone()
        p = _CountedProblem(self.problem)
        history = {key: [] for key in ['inner_grad_evals', 'inner_phase', 'K_t', 'inner_steps',
                   'inner_residual', 'grad_norm', 'lambda_min_H', 'radius', 'regularization',
                   'multiplier', 'step_norm', 'model_value', 'terminal', 'tr_kkt']}
        history.update(subproblem_type='trust_region', L1=self.L1, LH=self.LH, L2=self.L2,
                       sigma=self.sigma, radius_scale=self.radius_scale,
                       epsilon_g=self.epsilon_g, epsilon_H=self.epsilon_H,
                       inner_distance_target=self.A, n_updates=0, y_matches_output_x=False)
        solves = 0

        def finish(converged, reason):
            history['termination_reason'] = reason
            history['n_oracle_builds'] = len(history['grad_norm'])
            return AlgorithmResult(x=x, y=y, n_iterations=solves,
                                   n_inner_grad_evals=p.calls, n_subproblem_solves=solves,
                                   converged=converged, history=history)
        try:
            initial = p.grad_y(x, y)
            history['inner_grad_evals'].append(1)
            history['inner_phase'].append('initial_distance_bound')
            bound = initial.norm().item()/self.mu
            for _ in range(self.max_iterations):
                if not math.isfinite(bound):
                    raise RuntimeError('Non-finite inner distance bound')
                count = self._inner_count(bound)
                if count > self.max_inner_steps:
                    return finish(False, 'Required inner steps exceed max_inner_steps')
                before = p.calls
                try:
                    inner = self.inner_solver.run(p, x, y, stop_rule='steps', target=count)
                finally:
                    history['inner_grad_evals'].append(p.calls-before)
                    history['inner_phase'].append('fixed_count_agd')
                if inner.n_steps != count or inner.n_grad_evals != p.calls-before:
                    raise RuntimeError('Inner solver count contract failed')
                if not torch.isfinite(inner.y).all():
                    raise RuntimeError('Non-finite inner solution')
                y = inner.y.detach()
                history['y_matches_output_x'] = True
                g = p.grad_x(x, y)
                Hxx, Hxy, Hyx, Hyy = p.hessian_blocks(x, y)
                H = Hxx - Hxy @ torch.linalg.solve(Hyy, Hyx)
                H = (H+H.T)/2
                if not torch.isfinite(g).all() or not torch.isfinite(H).all():
                    raise RuntimeError('Non-finite raw oracle')
                gnorm = g.norm().item()
                radius = self.radius_scale*math.sqrt(max(gnorm, self.epsilon))
                shift = self.sigma*math.sqrt(gnorm)  # No epsilon floor here.
                B = H + shift*torch.eye(g.numel(), dtype=g.dtype, device=g.device)
                for key, value in [('K_t', count), ('inner_steps', inner.n_steps),
                                   ('inner_residual', inner.residual), ('grad_norm', gnorm),
                                   ('lambda_min_H', torch.linalg.eigvalsh(H)[0].item()),
                                   ('radius', radius), ('regularization', shift)]:
                    history[key].append(value)
                solves += 1
                sub = self.trust_region_solver.solve(g=g, H=B, radius=radius)
                metrics = self._check_tr(g, B, radius, sub)
                lam = float(sub.multiplier)
                terminal = gnorm <= self.epsilon and lam <= math.sqrt(self.L2*self.epsilon)
                for key, value in [('multiplier', lam), ('step_norm', sub.s.norm().item()),
                                   ('model_value', sub.model_value), ('terminal', terminal),
                                   ('tr_kkt', metrics)]:
                    history[key].append(value)
                if terminal:
                    return finish(True, 'GRTR Lemma 2.6 satisfied; returned current x_t')
                candidate = (x+sub.s).detach()
                if not torch.isfinite(candidate).all() or torch.equal(candidate, x):
                    return finish(False, 'Non-finite or stagnating outer update')
                x = candidate
                history['n_updates'] += 1
                history['y_matches_output_x'] = False
                bound = self.A + self.kappa*sub.s.norm().item()
            return finish(False, 'Maximum number of GRTR subproblems reached')
        except (RuntimeError, ValueError, OverflowError) as exc:
            return finish(False, str(exc))
