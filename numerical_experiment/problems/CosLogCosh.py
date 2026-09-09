import torch
from .NCSC import NCSCProblem

class CosLogCosh(NCSCProblem):

    def __init__(
        self,
        mu_y: float,
        omega: float,
        Q: torch.Tensor,
        beta: float,
    ):
        super().__init__()

        self.mu_y = float(mu_y)
        self.omega = float(omega)
        self.Q = Q
        self.beta = float(beta)

        self.a_mu = 0.5 * self.mu_y ** 0.5

    def f(self, x, y):

        term1 = torch.sum(
            (torch.cos(self.omega * x) - 1.0)
            / self.omega**2
        )

        term2 = (
            self.a_mu
            * (x @ self.Q @ y)
        )

        term3 = (
            -0.5
            * self.mu_y
            * torch.sum(y**2)
        )

        term4 = (
            -self.beta
            * torch.sum(torch.log(torch.cosh(y)))
        )

        return term1 + term2 + term3 + term4

    def value_remainder_y(self, x, y, z, grad_y, f_y, f_z):
        d = z - y
        a = torch.where(y >= 0, -2.0 * d, 2.0 * d)
        log_q = -torch.nn.functional.softplus(2.0 * torch.abs(y))
        q = torch.exp(log_q)

        small = torch.abs(a) < 1e-2
        u = torch.where(small, a, torch.zeros_like(a))
        w = q * torch.expm1(u)
        exp_remainder = u.square() * (
            0.5 + u * (1.0 / 6.0 + u * (1.0 / 24.0 + u * (
                1.0 / 120.0 + u * (1.0 / 720.0 + u / 5040.0)
            )))
        )
        log_remainder = w.square() * (
            -0.5 + w * (1.0 / 3.0 + w * (-0.25 + w * (
                0.2 + w * (-1.0 / 6.0 + w * (1.0 / 7.0 + w * (-0.125 + w / 9.0)))
            )))
        )
        small_remainder = q * exp_remainder + log_remainder
        large_remainder = torch.logaddexp(torch.log1p(-q), log_q + a) - q * a
        logcosh_remainder = torch.where(small, small_remainder, large_remainder)

        return -0.5 * self.mu_y * d.square().sum() - self.beta * logcosh_remainder.sum()

    def _calculate_constants(self):

        Q_norm = torch.linalg.matrix_norm(
            self.Q,
            ord=2,
        ).item()

        mu = self.mu_y

        ell = (
            max(
                1.0,
                self.mu_y + self.beta,
            )
            + self.a_mu * Q_norm
        )

        rho = max(
            self.omega,
            4.0 * self.beta
            / (3.0 * (3.0 ** 0.5)),
        )

        return {
            "ell": ell,
            "mu": mu,
            "rho": rho,
        }
