import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mpmath as mp
import torch

from problems.CosLogCosh import CosLogCosh


def main():
    torch.set_default_dtype(torch.float64)
    mp.mp.dps = 100
    problem = CosLogCosh(mu_y=1.0, omega=1.0, Q=torch.ones(1, 1), beta=0.5)
    x = torch.tensor([0.3])
    count = 0
    max_relative_error = 0.0
    for value in (0.0, 1e-6, 0.1, 1.0, -1.0, 10.0, -10.0, 40.0, -40.0, 100.0):
        for step in (0.0, 1e-12, 1e-8, 0.0049, 0.005, 0.0051, 0.01, 1.0, -1.0, 20.0, -20.0):
            y = torch.tensor([value])
            z = y + step
            actual = -problem.value_remainder_y(x, y, z, None, None, None).item()
            yy, zz = mp.mpf(y.item()), mp.mpf(z.item())
            d = zz - yy
            expected = float(d**2 / 2 + mp.mpf('0.5') * (
                mp.log(mp.cosh(zz)) - mp.log(mp.cosh(yy)) - mp.tanh(yy) * d
            ))
            if expected == 0:
                assert actual == 0
            else:
                relative_error = abs(actual - expected) / abs(expected)
                max_relative_error = max(max_relative_error, relative_error)
                assert relative_error < 2e-10, (value, step, actual, expected, relative_error)
            count += 1

    capture = Path('/tmp/scar_first_false_rejection.json')
    if capture.exists():
        data = json.loads(capture.read_text())
        problem = CosLogCosh(mu_y=data['mu_y'], omega=data['omega'],
                             Q=torch.tensor(data['Q']), beta=data['beta'])
        y, z = torch.tensor(data['v']), torch.tensor(data['y_trial'])
        lhs = -problem.value_remainder_y(torch.tensor(data['x']), y, z, None, None, None).item()
        rhs = (0.5 * data['L'] * (z - y).square().sum()).item()
        assert lhs <= rhs
        print(f"Captured rejection now passes: lhs={lhs:.16e}, rhs={rhs:.16e}")
    print(f"PASS: {count} comparisons with 100-digit reference; max relative error={max_relative_error:.6e}")


if __name__ == '__main__':
    main()
