import torch

def f(x, A, b):
    return 0.5 * x @ A @ x + b @ x

def gradient_descent(x0, A, b, lr = 0.01, n_steps = 100):
    x = x0.clone().detach()

    for _ in range(n_steps):
        x.requires_grad_(True)
        value = f(x, A, b)

        grad_x, = torch.autograd.grad(value, x)

        with torch.no_grad():
            x = x - lr * grad_x
    return x

if __name__ == "__main__":
    torch.manual_seed(0)
    n = 3
    A = torch.randn(n, n)
    A = A @ A.T + 1e-2 * torch.eye(n)
    b = torch.randn(n)

    x0 = torch.zeros(n, requires_grad=True)

    x_opt = gradient_descent(x0, A, b, lr=0.1, n_steps=10000)

    print("Optimal solution:", x_opt)