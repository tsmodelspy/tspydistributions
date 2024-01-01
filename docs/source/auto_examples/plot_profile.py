"""
Demo: Profile Distribution
--------------------------
"""

from tspydistributions.api import Distribution
from tspydistributions.pdqr import rsstd
from plotnine import ggtitle, coord_cartesian
x = rsstd(1000, mu = 0, sigma = 1, skew = 1, shape = 5)
d = Distribution(name = 'sstd')
f = d.estimate(x, type = 'FD')
s = d.profile(sim = 50, size = [100, 200, 400, 800, 1000, 2000], num_workers = 4)

s.summary()

print(s.plot(parameter='skew') + ggtitle('Profiled MSE for skew parameter') + coord_cartesian(ylim = (0.5, 2.0)))
print(s.plot(parameter='shape') + ggtitle('Profiled MSE for shape parameter') + coord_cartesian(ylim = (3, 20)))



