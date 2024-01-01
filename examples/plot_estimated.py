"""
Demo: API
---------
"""
from tspydistributions import pdqr
from tspydistributions import api
import numpy as np

# Estimation and Summary
# lamda = -0.5  == NIG distribution
x = pdqr.rsgh(2000, mu = 0, sigma = 1, skew = 0.8, shape = 4, lamda = -0.5)
d = api.Distribution(name = "sgh")
f = d.estimate(x, fixed = {'lamda':-0.5})
f.summary()

print(f.plot(type = 'density'))
print(f.plot(type = 'qq'))

# Expectation: 5% Risk Measures
value_at_risk = f.quantile(0.05)
expected_tail_loss = f.expectation(fun_str = "np.abs(1)", type = "q", lower = 0, upper = 0.05)/0.05
print(f'Value at Risk (5%): {round(value_at_risk[0], 4)}')
print(f'Expected Tail Loss (5%): {round(expected_tail_loss, 4)}')

