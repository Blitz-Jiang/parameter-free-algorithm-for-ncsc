import torch

from abc import ABC, abstractmethod


class NCSCProblem(ABC):

    def __init__(self):
        # Lazy cache:
        # {"ell": ..., "mu": ..., "rho": ...}
        self._constants = None

    # ============================================================
    # Objective
    # ============================================================

    @abstractmethod
    def f(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def value_remainder_y(self, x, y, z, grad_y, f_y, f_z):
        """Evaluate f(x,z) - f(x,y) - grad_y @ (z-y), without new oracle calls."""
        return f_z - f_y - grad_y @ (z - y)

    # ============================================================
    # Regularity constants
    # ============================================================

    @abstractmethod
    def _calculate_constants(self) -> dict:
        """
        Return valid global regularity constants:

        {
            "ell": gradient Lipschitz constant of f,
            "mu":  strong concavity constant in y,
            "rho": Hessian Lipschitz constant of f,
        }
        """
        pass

    def _get_constants(self):

        if self._constants is None:

            constants = self._calculate_constants()

            ell = float(constants["ell"])
            mu = float(constants["mu"])
            rho = float(constants["rho"])

            if ell <= 0:
                raise ValueError("ell must be positive.")

            if mu <= 0:
                raise ValueError("mu must be positive.")

            if rho <= 0:
                raise ValueError("rho must be positive.")

            if mu > ell:
                raise ValueError("Require mu <= ell.")

            self._constants = {
                "ell": ell,
                "mu": mu,
                "rho": rho,
            }

        return self._constants

    @property
    def ell(self) -> float:
        return self._get_constants()["ell"]

    @property
    def mu(self) -> float:
        return self._get_constants()["mu"]

    @property
    def rho(self) -> float:
        return self._get_constants()["rho"]

    @property
    def kappa(self) -> float:
        return self.ell / self.mu

    # ============================================================
    # Autograd derivatives
    # ============================================================

    def grad_y(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:

        y_var = (
            y.clone()
            .detach()
            .requires_grad_(True)
        )

        value = self.f(x, y_var)

        grad_y, = torch.autograd.grad(
            value,
            y_var,
        )

        return grad_y.detach()

    def grad_x(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:

        x_var = (
            x.clone()
            .detach()
            .requires_grad_(True)
        )

        value = self.f(x_var, y)

        grad_x, = torch.autograd.grad(
            value,
            x_var,
        )

        return grad_x.detach()

    def hessian_blocks(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):

        x_ = x.clone().detach()
        y_ = y.clone().detach()

        H = torch.autograd.functional.hessian(
            lambda xx, yy: self.f(xx, yy),
            (x_, y_),
        )

        H_xx = H[0][0]
        H_xy = H[0][1]
        H_yx = H[1][0]
        H_yy = H[1][1]

        return (
            H_xx.detach(),
            H_xy.detach(),
            H_yx.detach(),
            H_yy.detach(),
        )
