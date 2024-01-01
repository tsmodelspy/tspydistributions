from pytest import approx
import tspydistributions.pdqr as dist
from tspydistributions.api import Distribution

def test_api_norm_expectation():
    d = Distribution(name = 'norm')
    assert approx(d.expectation('np.abs(x)'), 0.01) == 0.797

def test_api_jsu_expectation():
    d = Distribution(name = 'jsu', shape = 1)
    assert approx(d.expectation('np.abs(x)'), 0.01) == 0.60163

def test_api_moments(name = 'sstd', skew = 0.5, shape = 6.1, lamda = -0.5):
    d = Distribution(name = name, skew = skew, shape = shape, lamda = lamda)
    mu = d.expectation('x')
    sigma = d.expectation('(x-0.0)**2') ** 0.5
    skewness = d.expectation('((x-0.0)/1.0)**3')
    kurtosis = d.expectation('((x-0.0)/1.0)**4')/(1.0**4)
    cond_mu = round(mu, 5) == 0.0
    cond_sigma = approx(sigma, 0.01) == 1.0
    cond_skewness = approx(skewness, 0.01) == d.skewness()[0]
    cond_kurtosis = approx(kurtosis, 0.01) == d.kurtosis()[0]
    conditions = [cond_mu, cond_sigma, cond_skewness, cond_kurtosis]
    assert all(conditions)

