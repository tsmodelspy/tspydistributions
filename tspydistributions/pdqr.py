from __future__ import annotations
from tspydistributions.helpers import _validate_distribution
import numpy as np
from scipy.special import gamma, beta, gammaln, kve
from scipy.stats import gamma as gammadist
from scipy.stats import t, norm, uniform, genhyperbolic
from typing import List, Literal, Any, Optional, TYPE_CHECKING, TypeVar
from scipy.integrate import quad
from scipy.optimize import minimize, brentq
import numpy.typing as npt
import inspect

Vector = npt.ArrayLike
Array = npt.NDArray[np.float64]

def heaviside(x: Vector)-> Array:
    x = np.atleast_1d(x)
    return (np.sign(x) + 1.0)/2.0

def signum(x: Vector)-> Array:
    x = np.atleast_1d(x)
    f = lambda x: (x >= 0).astype(np.float64) - (x < 0).astype(np.float64)
    return np.fromiter((f(x) for x in x), dtype = np.float64)

def scaler(x: Vector, mu: Vector, sigma: Vector) -> Vector:
    return np.subtract(x,mu)/sigma

def fs_skew_moments(m: Vector, skew: Vector) -> Array:
    m_squared = np.power(m, 2.0)
    skew_squared = np.power(skew, 2.0)
    mu = m * np.subtract(skew, np.divide(1.0,skew))
    sigma = np.sqrt((1 - m_squared) * (skew_squared + 1.0/skew_squared) + 2.0 * m_squared - 1.0)
    out = np.asarray([mu, sigma])
    return out

def kappagh(x: Vector, lamda: Vector = 0.5)-> Array:
    if lamda ==  -0.5:
        kappa = np.divide(1.0,x)
    else:
        kappa = (kve(np.add(lamda,1.0), x)/kve(lamda, x))/x
    return kappa

def deltakappagh(x: Vector, lamda: Vector = 0.5)-> Array:
    if lamda == -0.5:
        deltakappa = np.subtract(kappagh(x, np.add(lamda, 1.0)), kappagh(x, lamda))
    else:
        deltakappa = np.subtract(kappagh(x, np.add(lamda,1.0)), kappagh(x, lamda))
    return deltakappa
 
def paramgh(rho: Vector, zeta: Vector, lamda: Vector)-> Array:
    rho2 = 1 - np.power(rho, 2.0)
    alpha = np.power(zeta, 2.0) * kappagh(zeta, lamda)/rho2
    alpha = np.sqrt(alpha * (1.0 + np.multiply(np.power(rho, 2.0), np.power(zeta, 2.0)) * deltakappagh(zeta, lamda)/rho2))
    beta = alpha * rho
    delta = np.divide(zeta, (alpha * np.sqrt(rho2)))
    mu = np.multiply(np.multiply(-1.0, beta), np.power(delta, 2.0)) * kappagh(zeta, lamda)
    params = np.asarray([alpha, beta, delta, mu, lamda], dtype = np.float64)
    return params

def paramghst(betabar: Vector, nu: Vector)-> Array:
    delta = np.sqrt(1.0/(((2.0 * np.power(betabar,2.0))/(np.power(np.subtract(nu,2.0), 2.0) * np.subtract(nu,4.0))) + (1.0/np.subtract(nu,2.0))))
    beta = betabar / delta
    mu = -1.0 * ((beta * np.power(delta, 2.0)) / np.subtract(nu,2.0))
    params =  np.asarray([mu, delta, beta, nu], dtype = np.float64)
    return params

def paramghconvert(params: Vector)-> Array:
    params = np.asarray(params, dtype = np.float64)
    alpha = params[0] * params[2]
    beta = params[1] * params[2]
    loc = params[3]
    scale = params[2]
    lamda = params[4]
    return np.asarray([alpha, beta, loc, scale, lamda], dtype = np.float64)


def dstd(x: Vector, mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5, log: bool = False) -> Array:
    """
    (Student) Probability Density Function (dstd)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to x 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param x: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param shape: the shape parameter (degrees of freedom)
    :param log: whether to return the log density
    :rtype: a numpy array
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, shape]):
        v_fun = np.vectorize(dstd)
        return v_fun(x, mu = mu, sigma = sigma, shape = shape, log = log)
    
    parameters = np.array([mu, sigma, shape], dtype = np.float64)
    x = scaler(x, parameters[0], parameters[1])
    scale = np.sqrt(parameters[2]/(parameters[2]-2.0))
    x = x * scale
    alpha = gamma((parameters[2] + 1.0)/2.0)/np.sqrt(np.pi * parameters[2])
    beta = gamma(parameters[2]/2.0) * np.power((1.0 + np.square(x)/parameters[2]), (parameters[2] + 1.0)/2.0)
    ratio = alpha/beta
    pdf = (scale * ratio)/parameters[1]
    if log:
        pdf = np.log(pdf)
    return pdf

def pstd(q: Vector, mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5, lower_tail: bool = True) -> Array:
    """
    (Student) Cumulative Probability Function (pstd)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to q 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param q: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param shape: the shape parameter (degrees of freedom)
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    q = np.atleast_1d(q)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, shape]):
        v_fun = np.vectorize(pstd)
        return v_fun(q, mu = mu, sigma = sigma, shape = shape, lower_tail = lower_tail)
        
    parameters = np.array([mu, sigma, shape], dtype = np.float64)
    scale = np.sqrt(parameters[2]/(parameters[2]-2.0))
    q = scaler(q, parameters[0],parameters[1])
    p = np.atleast_1d(t.cdf(q * scale, parameters[2]))
    if lower_tail == False:
        p = 1.0 - p
    return p

def qstd(p: Vector, mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5, lower_tail: bool = True) -> Array:
    """
    (Student) Quantile Function (qstd)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to p 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param shape: the shape parameter (degrees of freedom)
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    p = np.atleast_1d(p)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, shape]):
        v_fun = np.vectorize(qstd)
        return v_fun(p, mu = mu, sigma = sigma, shape = shape, lower_tail = lower_tail)
    
    parameters = np.array([mu,sigma,shape], dtype = np.float64)
    if lower_tail == False:
        p = 1.0 - p
    scale = np.sqrt(parameters[2]/(parameters[2] - 2.0))
    q = np.atleast_1d(t.ppf(p, parameters[2]) * parameters[1]/scale + parameters[0])
    return q

def rstd(n: int = 1, mu: float = 0, sigma: float = 1, shape: float = 5, seed: Optional[int] = None) -> Array:
    """
    (Student) Random Number Function (rstd)
    
    :param n: the number of draws
    :param mu: the mean
    :param sigma: the standard deviation
    :param shape: the shape parameter (degrees of freedom)
    :param seed: an optional value to initialize the random seed generator 
    :rtype: a numpy array
    """
    parameters = np.asarray([mu,sigma,shape], dtype = np.float64)
    scale = np.sqrt(parameters[2]/(parameters[2] - 2.0))
    r = t.rvs(parameters[2], size = n, random_state = seed) * 1.0 / scale
    return r

def dnorm(x: Vector, mu: Vector = 0, sigma: Vector = 1, log: bool = False) -> Array:
    """
    (Normal) Probability Density Function (dnorm)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to x 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param x: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param log: whether to return the log density

    :rtype: a numpy array
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    if any(param.size > 1 for param in [mu, sigma]):
        v_fun = np.vectorize(dnorm)
        return v_fun(x, mu = mu, sigma = sigma, log = log)
    
    parameters = np.array([mu, sigma], dtype = np.float64)
    pdf = norm.pdf(x, loc = parameters[0], scale = parameters[1])
    if log:
        pdf = np.log(pdf)
    return pdf

def pnorm(q: Vector, mu: Vector = 0, sigma: Vector = 1, lower_tail: bool = True) -> Array:
    """
    (Normal) Cumulative Probability Function (pnorm)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to q 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param q: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    q = np.atleast_1d(q)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    if any(param.size > 1 for param in [mu, sigma]):
        v_fun = np.vectorize(pnorm)
        return v_fun(q, mu = mu, sigma = sigma, lower_tail = lower_tail)
    
    parameters = np.array([mu, sigma], dtype = np.float64)
    p = norm.cdf(q, loc = parameters[0], scale = parameters[1])
    if lower_tail == False:
        p = 1.0 - p
    return p

def qnorm(p: Vector, mu: Vector = 0, sigma: Vector = 1, lower_tail: bool = True) -> Array:
    """
    (Normal) Quantile Function (qnorm)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to p 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    p = np.atleast_1d(p)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    if any(param.size > 1 for param in [mu, sigma]):
        v_fun = np.vectorize(qnorm)
        return v_fun(p, mu = mu, sigma = sigma, lower_tail = lower_tail)
    
    parameters = np.array([mu,sigma], dtype = np.float64)
    if lower_tail == False:
        p = 1.0 - p
    q = norm.ppf(p, loc = parameters[0], scale = parameters[1])
    return q

def rnorm(n: int = 1, mu: float = 0, sigma: float = 1, seed: Optional[int] = None) -> Array:
    """
    (Normal) Random Number Function (rnorm)

    :param n: the number of draws
    :param mu: the mean
    :param sigma: the standard deviation
    :param seed: an optional value to initialize the random seed generator 
    :rtype: a numpy array
    """
    parameters = np.asarray([mu,sigma], dtype = np.float64)
    r = norm.rvs(loc = parameters[0], scale = parameters[1], size = n, random_state = seed)
    return np.atleast_1d(r)


def dged(x: Vector, mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5, log: bool = False) -> Array:
    """
    (GED) Probability Density Function (dged)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to x 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param x: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param shape: the shape parameter
    :param log: whether to return the log density
    :rtype: a numpy array
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, shape]):   
        v_fun = np.vectorize(dged)
        return v_fun(x, mu = mu, sigma = sigma, shape = shape, log = log)
    
    parameters = np.array([mu, sigma, shape], dtype = np.float64)
    x = scaler(x, parameters[0], parameters[1])
    lam = np.sqrt(np.power(0.5, 2.0/parameters[2]) * gamma(1.0/parameters[2])/gamma(3.0/parameters[2]))
    g = parameters[2]/(lam * (np.power(2.0,1.0 + (1.0/parameters[2]))) * gamma(1.0/parameters[2]))
    pdf = g * np.exp(-0.5 * np.power(np.abs(x/lam), parameters[2]))
    pdf = pdf/parameters[1]
    if log:
        pdf = np.log(pdf)
    return pdf

def pged(q: Vector, mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5, lower_tail: bool = True) -> Array:
    """
    (GED) Cumulative Probability Function (pged)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to q 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param q: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param shape: the shape parameter
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    q = np.atleast_1d(q)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, shape]):   
        v_fun = np.vectorize(pged)
        return v_fun(q, mu = mu, sigma = sigma, shape = shape, lower_tail = lower_tail)
    
    parameters = np.array([mu, sigma, shape], dtype = np.float64)
    q = scaler(q, parameters[0], parameters[1])
    lam = np.sqrt(1.0/np.power(2.0, (2.0/parameters[2])) * gamma(1.0/parameters[2]) / gamma(3.0/parameters[2]))
    g  = parameters[2]/(lam * (np.power(2.0,(1 + 1.0/parameters[2]))) * gamma(1.0/parameters[2]))
    h = np.power(2.0, (1.0/parameters[2])) * lam * g * gamma(1.0/parameters[2])/parameters[2]
    s = 0.5 * np.power(np.abs(q)/lam, parameters[2])
    p = 0.5 + np.sign(q) * h * gammadist.cdf(s, 1.0/parameters[2], scale = 1, loc = 0)
    if lower_tail == False:
        p = 1.0 - p
    return p

def qged(p: Vector, mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5, lower_tail: bool = True) -> Array:
    """
    (GED) Quantile Function (qged)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to p 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param shape: the shape parameter (degrees of freedom)
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    p = np.atleast_1d(p)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, shape]):
        v_fun = np.vectorize(qged)
        return v_fun(p, mu = mu, sigma = sigma, shape = shape, lower_tail = lower_tail)
    
    parameters = np.array([mu,sigma,shape], dtype = np.float64)
    if lower_tail == False:
        p = 1.0 - p
    p = 2.0 * p - 1.0
    lam = np.sqrt(1.0/np.power(2.0, (2.0/parameters[2])) * gamma(1.0/parameters[2]) / gamma(3.0/parameters[2]))
    q = lam * np.power(2.0 * gammadist.ppf(np.abs(p), 1.0/parameters[2], scale = 1, loc = 0), 1.0/parameters[2])
    q = q * np.sign(p)
    q = q * parameters[1] + parameters[0]
    return q

def rged(n: int = 1, mu: float = 0, sigma: float = 1, shape: float = 5, seed: Optional[int] = None) -> Array:
    """
    (GED) Random Number Function (rged)

    :param n: the number of draws
    :param mu: the mean
    :param sigma: the standard deviation
    :param shape: the shape parameter (degrees of freedom)
    :param seed: an optional value to initialize the random seed generator 
    :rtype: a numpy array
    """
    parameters = np.asarray([mu,sigma,shape], dtype = np.float64)
    lam = np.sqrt(np.power(0.5, 2.0/parameters[2]) * gamma(1.0/parameters[2])/gamma(3.0/parameters[2]))
    rgam = gammadist.rvs(1.0/parameters[2], scale = 1.0, loc = 0.0, size = n, random_state = seed)
    r = lam * np.power(2.0 * rgam, 1.0/parameters[2]) * np.sign(uniform.rvs(size = n, random_state = seed) - 0.5)
    r = parameters[0] + r * parameters[1]
    return r

def dsnorm(x: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, log: bool = False) -> Array:
    """
    (Skew-Normal) Probability Density Function (dsnorm)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to x 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param x: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param log: whether to return the log density
    :rtype: a numpy array
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)

    if any(param.size > 1 for param in [mu, sigma, skew]):
        v_fun = np.vectorize(dsnorm)
        return v_fun(x, mu = mu, sigma = sigma, skew = skew, log = log)
    
    parameters = np.array([mu, sigma, skew], dtype = np.float64)
    x = scaler(x, parameters[0], parameters[1])
    m = np.divide(2.0,np.sqrt(2.0 * np.pi))
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    z = x * fs_sigma + fs_mu
    z_sign = np.sign(z)
    xi = np.power(parameters[2], z_sign)
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    pdf = g * dnorm(z/xi) * fs_sigma
    pdf = pdf/parameters[1]
    if log:
        pdf = np.log(pdf)
    return pdf

def psnorm(q: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, lower_tail: bool = True) -> Array:
    """
    (Skew-Normal) Cumulative Probability Function (psnorm)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to q 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param q: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter 
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    q = np.atleast_1d(q)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)

    if any(param.size > 1 for param in [mu, sigma, skew]):
        v_fun = np.vectorize(psnorm)
        return v_fun(q, mu = mu, sigma = sigma, skew = skew, lower_tail = lower_tail)

    parameters = np.array([mu, sigma, skew], dtype=np.float64)
    q = scaler(q, parameters[0], parameters[1])
    m = 2.0/np.sqrt(2.0 * np.pi)
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    z = q * fs_sigma + fs_mu
    z_sign = signum(z)
    xi = np.power(parameters[2], z_sign)
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    p = heaviside(z) - z_sign * g * xi * pnorm(-np.abs(z)/xi)
    if lower_tail == False:
        p = 1.0 - p
    return p

def qsnorm(p: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, lower_tail: bool = True) -> Array:
    """
    (Skew-Normal) Quantile Function (qsnorm)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to p 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter 
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    p = np.atleast_1d(p)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    
    if any(param.size > 1 for param in [mu, sigma, skew]):
        v_fun = np.vectorize(qsnorm)
        return v_fun(p, mu = mu, sigma = sigma, skew = skew, lower_tail = lower_tail)

    parameters = np.array([mu,sigma,skew], dtype = np.float64)
    if lower_tail == False:
        p = 1.0 - p
    m = 2.0/np.sqrt(2.0 * np.pi)
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    z = p - (1.0/(1.0 + np.power(parameters[2], 2.0)))
    z_sign = np.sign(z)
    xi = np.power(parameters[2], z_sign)
    tmp = (heaviside(z) - z_sign * p)/(g * xi)
    fun = lambda x, s: qnorm(x, mu = 0, sigma = s)[0]
    v_qnorm = np.vectorize(fun, otypes=[np.float64])
    q = v_qnorm(tmp, xi)
    q = (-1.0 * z_sign * q - fs_mu)/fs_sigma
    q = parameters[0] + q * parameters[1]
    return q

def rsnorm(n: int = 1, mu: float = 0, sigma: float = 1, skew: float = 2, seed: Optional[int] = None) -> Array:
    """
    (Skew-Normal) Random Number Function (rsnorm)

    :param n: the number of draws
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param seed: an optional value to initialize the random seed generator 
    :rtype: a numpy array
    """
    parameters = np.asarray([mu,sigma,skew], dtype = np.float64)
    weight = parameters[2]/(parameters[2] + 1/parameters[2])
    z = uniform.rvs(size = n,loc = -1.0 * weight, scale = weight + (1.0 - weight), random_state = seed)
    z_sign = np.sign(z)
    xi = np.power(parameters[2], z_sign)
    r = -1.0 * np.abs(rnorm(n = n, seed = seed))/xi * z_sign
    m = 2.0/np.sqrt(2.0 * np.pi)
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    r = (r - fs_mu)/fs_sigma
    r = parameters[0] + r * parameters[1]
    return r

def dsged(x: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, log: bool = False) -> Array:
    """
    (Skew-GED) Probability Density Function (dsged)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to x 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param x: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param log: whether to return the log density
    :rtype: a numpy array
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(dsged)
        return v_fun(x, mu = mu, sigma = sigma, skew = skew, shape = shape, log = log)

    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    x = scaler(x, parameters[0], parameters[1])
    lam = np.sqrt(np.power(2.0,-2.0/parameters[3]) * gamma(1.0/parameters[3])/gamma(3.0/parameters[3]))
    m = np.power(2.0, (1.0/parameters[3])) * lam * gamma(2.0/parameters[3])/gamma(1.0/parameters[3])
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    z = x * fs_sigma + fs_mu
    z_sign = np.sign(z)
    xi = np.power(parameters[2], z_sign)
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    pdf = g * dged(z/xi, shape = parameters[3]) * fs_sigma
    pdf = pdf/parameters[1]
    if log:
        pdf = np.log(pdf)
    return pdf

def psged(q: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lower_tail: bool = True) -> Array:
    """
    (Skew-GED) Cumulative Probability Function (psged)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to q 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param q: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    q = np.atleast_1d(q)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(psged)
        return v_fun(q, mu = mu, sigma = sigma, skew = skew, shape = shape, lower_tail = lower_tail)

    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    q = scaler(q, parameters[0], parameters[1])
    lam = np.sqrt(np.power(2.0,-2.0/parameters[3]) * gamma(1.0/parameters[3])/gamma(3.0/parameters[3]))
    m = np.power(2.0, (1.0/parameters[3])) * lam * gamma(2.0/parameters[3])/gamma(1.0/parameters[3])
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    z = q * fs_sigma + fs_mu
    z_sign = signum(z)
    xi = np.power(parameters[2], z_sign)
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    p = heaviside(z) - z_sign * g * xi * pged(-np.abs(z)/xi, 0., 1., parameters[3])
    if lower_tail == False:
        p = 1.0 - p
    return p


def qsged(p: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lower_tail: bool = True) -> Array:
    """
    (Skew-GED) Quantile Function (qsged)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to p 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    p = np.atleast_1d(p)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(qsged)
        return v_fun(p, mu = mu, sigma = sigma, skew = skew, shape = shape, lower_tail = lower_tail)

    if lower_tail == False:
        p = 1.0 - p
    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    lam = np.sqrt(np.power(2.0,-2.0/parameters[3]) * gamma(1.0/parameters[3])/gamma(3.0/parameters[3]))
    m = np.power(2.0, (1.0/parameters[3])) * lam * gamma(2.0/parameters[3])/gamma(1.0/parameters[3])
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    z = p - (1.0/(1.0 + np.power(parameters[2], 2.0)))
    z_sign = np.sign(z)
    xi = np.power(parameters[2], z_sign)
    tmp = (heaviside(z) - z_sign * p)/(g * xi)
    q = (-1.0 * z_sign * qged(tmp, 0., 1., parameters[3]) * xi - fs_mu)/fs_sigma
    q = parameters[0] + q * parameters[1]
    return q

def rsged(n: int = 1, mu: float = 0, sigma: float = 1, skew: float = 2, shape: float = 4, seed: Optional[int] = None) -> Array:
    """
    (Skew-GED) Random Number Function (rsged)

    :param n: the number of draws
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param seed: an optional value to initialize the random seed generator 
    :rtype: a numpy array
    """    
    parameters = np.asarray([mu,sigma,skew,shape], dtype = np.float64)
    weight = parameters[2]/(parameters[2] + 1/parameters[2])
    z = uniform.rvs(size = n,loc = -1.0 * weight, scale = weight + (1.0 - weight), random_state = seed)
    z_sign = np.sign(z)
    xi = np.power(parameters[2], z_sign)
    r = (-1.0 * np.abs(rged(n = n, shape = parameters[3], seed = seed)))/xi * z_sign
    lam = np.sqrt(np.power(2.0,-2.0/parameters[3]) * gamma(1.0/parameters[3])/gamma(3.0/parameters[3]))
    m = np.power(2.0, (1.0/parameters[3])) * lam * gamma(2.0/parameters[3])/gamma(1.0/parameters[3])
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    r = (r - fs_mu)/fs_sigma
    r = parameters[0] + r * parameters[1]
    return r

def dsstd(x: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, log: bool = False) -> Array:
    """
    (Skew-Student) Probability Density Function (dsstd)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to x 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param x: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param log: whether to return the log density
    :rtype: a numpy array
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(dsstd)
        return v_fun(x, mu = mu, sigma = sigma, skew = skew, shape = shape, log = log)

    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    x = scaler(x, parameters[0], parameters[1])
    m = 2.0 * np.sqrt(parameters[3] - 2.0)/(parameters[3] - 1.0)/beta(0.5, parameters[3]/2.0)
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    z = x * fs_sigma + fs_mu
    z_sign = np.sign(z)
    xi = np.power(parameters[2], z_sign)
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    pdf = g * dstd(z/xi, shape = parameters[3]) * fs_sigma
    pdf = pdf/parameters[1]
    if log:
        pdf = np.log(pdf)
    return pdf

def psstd(q: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lower_tail: bool = True) -> Array:
    """
    (Skew-Student) Cumulative Probability Function (psstd)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to q 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param q: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    q = np.atleast_1d(q)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(psstd)
        return v_fun(q, mu = mu, sigma = sigma, skew = skew, shape = shape, lower_tail = lower_tail)

    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    q = scaler(q, parameters[0], parameters[1])
    m = 2.0 * np.sqrt(parameters[3] - 2.0)/(parameters[3] - 1.0)/beta(0.5, parameters[3]/2.0)
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    z = q * fs_sigma + fs_mu
    z_sign = signum(z)
    xi = np.power(parameters[2], z_sign)
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    p = heaviside(z) - z_sign * g * xi * pstd(-np.abs(z)/xi, shape = parameters[3])
    if lower_tail == False:
        p = 1.0 - p
    return p

def qsstd(p: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lower_tail: bool = True) -> Array:
    """
    (Skew-Student) Quantile Function (qsstd)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to p 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    p = np.atleast_1d(p)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(qsstd)
        return v_fun(p, mu = mu, sigma = sigma, skew = skew, shape = shape, lower_tail = lower_tail)

    if lower_tail == False:
        p = 1.0 - p
    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    m = 2.0 * np.sqrt(parameters[3] - 2.0)/(parameters[3] - 1.0)/beta(0.5, parameters[3]/2.0)
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    z = p - (1.0/(1.0 + np.power(parameters[2], 2.0)))
    z_sign = np.sign(z)
    xi = np.power(parameters[2], z_sign)
    tmp = (heaviside(z) - z_sign * p)/(g * xi)
    q = (-1.0 * z_sign * qstd(tmp, mu = 0, sigma = xi, shape = parameters[3]) - fs_mu)/fs_sigma
    q = parameters[0] + q * parameters[1]
    return q

def rsstd(n: int = 1, mu: float = 0, sigma: float = 1, skew: float = 2, shape: float = 4, seed: Optional[int] = None) -> Array:
    """
    (Skew-Student) Random Number Function (rsstd)

    :param n: the number of draws
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param seed: an optional value to initialize the random seed generator 
    :rtype: a numpy array
    """
    parameters = np.asarray([mu,sigma,skew,shape], dtype = np.float64)
    weight = parameters[2]/(parameters[2] + 1/parameters[2])
    z = uniform.rvs(size = n,loc = -1.0 * weight, scale = weight + (1.0 - weight), random_state = seed)
    z_sign = np.sign(z)
    xi = np.power(parameters[2], z_sign)
    r = (-1.0 * np.abs(rstd(n = n, shape = parameters[3], seed = seed)))/xi * z_sign
    m = 2.0 * np.sqrt(parameters[3] - 2.0)/(parameters[3] - 1.0)/beta(0.5, parameters[3]/2.0)
    fs_mu, fs_sigma = fs_skew_moments(m, parameters[2])
    r = (r - fs_mu)/fs_sigma
    r = parameters[0] + r * parameters[1]
    return np.atleast_1d(r)

def djsu(x: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, log: bool = False) -> Array:
    """
    (Johnson's SU) Probability Density Function (djsu)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to x 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param x: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param log: whether to return the log density
    :rtype: a numpy array
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(djsu)
        return v_fun(x, mu = mu, sigma = sigma, skew = skew, shape = shape, log = log)

    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    x = scaler(x, parameters[0], parameters[1])
    rtau = 1.0/parameters[3]
    if rtau < 1e-8:
        w = 1.0
    else:
        w = np.exp(np.power(rtau, 2.0))
    omega = -1.0 * parameters[2] * rtau
    c = np.sqrt(1.0/(0.5 * (w - 1) * (w * np.cosh(2.0 * omega) + 1.0)))
    z = (x - (c * np.sqrt(w) * np.sinh(omega)))/c
    r = - 1.0 * parameters[2] + np.arcsinh(z)/rtau
    pdf = -1.0 * np.log(c) - np.log(rtau) - 0.5 * np.log(np.power(z,2.0) + 1.0) - 0.5 * np.log(2.0 * np.pi) - 0.5 * np.power(r, 2.0)
    pdf = np.exp(pdf)/parameters[1]
    if log == True:
        pdf = np.log(pdf)
    return np.atleast_1d(pdf)

def pjsu(q: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lower_tail: bool = True) -> Array:
    """
    (Johnson's SU) Cumulative Probability Function (pjsu)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to q 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param q: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    q = np.atleast_1d(q)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(pjsu)
        return v_fun(q, mu = mu, sigma = sigma, skew = skew, shape = shape, lower_tail = lower_tail)

    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    rtau = 1.0/parameters[3]
    if rtau < 1e-8:
        w = 1.0
    else:
        w = np.exp(np.power(rtau, 2.0))
    omega = -1.0 * parameters[2] * rtau
    c = 1.0/np.sqrt(0.5 * (w - 1.0) * (w * np.cosh(2.0 * omega) + 1.0))
    z = (q - (parameters[0] + c * parameters[1] * np.sqrt(w) * np.sinh(omega)))/(c * parameters[1])
    r = -1.0 * parameters[2] + np.arcsinh(z)/rtau
    p = pnorm(r)
    if lower_tail == False:
        p = 1.0 - p
    return p

def qjsu(p: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lower_tail: bool = True) -> Array:
    """
    (Johnson's SU) Quantile Function (qjsu)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to p 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    p = np.atleast_1d(p)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(qjsu)
        return v_fun(p, mu = mu, sigma = sigma, skew = skew, shape = shape, lower_tail = lower_tail)

    if lower_tail == False:
        p = 1.0 - p
    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    rtau = 1.0/parameters[3]
    if rtau < 1e-8:
        w = 1.0
    else:
        w = np.exp(np.power(rtau, 2.0))
    nq = qnorm(p)
    z = np.sinh(rtau * (nq + parameters[2]))
    omega = -1.0 * parameters[2] * rtau
    c = np.sqrt(1.0/(0.5 * (w - 1.0) * (w * np.cosh(2.0 * omega) + 1.0)))
    q = (c * np.sqrt(w) * np.sinh(omega)) + c * z
    q = parameters[0] + q * parameters[1]
    return q

def rjsu(n: int = 1, mu: float = 0, sigma: float = 1, skew: float = 2, shape: float = 4, seed: Optional[int] = None) -> Array:
    """
    (Johnson's SU) Random Number Function (rjsu)

    :param n: the number of draws
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param seed: an optional value to initialize the random seed generator 
    :rtype: a numpy array
    """
    parameters = np.asarray([mu,sigma,skew,shape], dtype = np.float64)
    x = uniform.rvs(size = n, random_state = seed)
    r = qjsu(x, skew = parameters[2], shape = parameters[3])
    r = parameters[0] + r * parameters[1]
    return r

def dgh(x: Vector, params: Vector) -> Array:
    """
    The non-stamdardized GH distribution
    params are [alpha, beta, delta, mu, lamda]
    """
    x = np.asarray(x, dtype = np.float64)
    params = np.asarray(params, dtype=np.float64)
    arg1 = params[2] * np.sqrt(np.power(params[0], 2.0) - np.power(params[1], 2.0))
    a = (params[4]/2.0) * np.log(np.power(params[0],2.0) - np.power(params[1],2.0))\
          - (np.log(np.sqrt(2.0 * np.pi)) + (params[4] - 0.5) * np.log(params[0])\
             + params[4] * np.log(params[2]) + np.log(kve(params[4], arg1)) - arg1)
    f = ((params[4] - 0.5)/2.0) * np.log(np.power(params[2],2.0) + np.power((x - params[3]),2.0))
    arg2 = params[0] * np.sqrt(np.power(params[2],2.0) + np.power((x - params[3]), 2.0))
    k = np.log(kve(params[4] - 0.5, arg2)) - arg2
    e = params[1] * (x - params[3])
    pdf = np.exp(a + f + k + e)
    return np.atleast_1d(pdf)

def pgh(q: Vector, params: Vector) -> Array:
    """
    (Generalized Hyperbolic) Cumulative Probability Function (pgh)
    """
    q = np.atleast_1d(q)
    params = np.asarray(params, dtype=np.float64)
    fdgh = lambda x: dgh(x, params)[0]
    p = np.nan * np.ones(len(q))
    p = np.fromiter((quad(fdgh, -np.inf, q)[0] for q in q), dtype = np.float64)
    return p

    
def dsgh(x: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lamda: Vector = 0, log: bool = False) -> Array:
    """
    (Standardized Generalized Hyperbolic) Probability Density Function (dsgh)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to x 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param x: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lamda: the second shape parameter of the GH distribution related to the GIG distribution.
    :param log: whether to return the log density
    :rtype: a numpy array
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    lamda = np.atleast_1d(lamda)

    if any(param.size > 1 for param in [mu, sigma, skew, shape, lamda]):
        v_fun = np.vectorize(dsgh)
        return v_fun(x, mu = mu, sigma = sigma, skew = skew, shape = shape, lamda = lamda, log = log)

    parameters = np.array([mu, sigma, skew, shape, lamda], dtype=np.float64)
    x = scaler(x, parameters[0], parameters[1])
    params = paramgh(parameters[2],parameters[3],parameters[4])
    pdf = dgh(x, params)/parameters[1]    
    if log == True:
        pdf = np.log(pdf)
    return np.atleast_1d(pdf)

def psgh(q: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lamda: Vector = 0, lower_tail: bool = True) -> Array:
    """
    (Standardized Generalized Hyperbolic) Cumulative Probability Function (psgh)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to q 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param q: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lamda: the second shape parameter of the GH distribution related to the GIG distribution.
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    q = np.atleast_1d(q)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    lamda = np.atleast_1d(lamda)

    if any(param.size > 1 for param in [mu, sigma, skew, shape, lamda]):
        v_fun = np.vectorize(psgh)
        return v_fun(q, mu = mu, sigma = sigma, skew = skew, shape = shape, lamda = lamda, lower_tail = lower_tail)

    parameters = np.array([mu, sigma, skew, shape, lamda], dtype=np.float64)
    q = scaler(q, parameters[0], parameters[1])
    params = paramgh(parameters[2], parameters[3], parameters[4])
    p = pgh(q, params)
    if lower_tail == False:
        p = 1.0 - p
    return np.atleast_1d(p)

def qsgh(p: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lamda: Vector = 0, lower_tail: bool = True) -> Array:
    """
    (Standardized Generalized Hyperbolic) Quantile Function (qsgh)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to p 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lamda: the second shape parameter of the GH distribution related to the GIG distribution.
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    p = np.atleast_1d(p)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    lamda = np.atleast_1d(lamda)

    if any(param.size > 1 for param in [mu, sigma, skew, shape, lamda]):
        v_fun = np.vectorize(qsgh)
        return v_fun(p, mu = mu, sigma = sigma, skew = skew, shape = shape, lamda = lamda, lower_tail = lower_tail)

    if lower_tail == False:
        p = 1.0 - p
    parameters = np.array([mu, sigma, skew, shape, lamda], dtype=np.float64)
    params = paramgh(parameters[2], parameters[3], parameters[4])
    params = paramghconvert(params)
    q = genhyperbolic.ppf(p, a = params[0], b = params[1], loc = params[2], scale = params[3], p = params[4])
    q = parameters[0] + q * parameters[1]
    return np.atleast_1d(q)

def rsgh(n: int = 1, mu: float = 0, sigma: float = 1, skew: float = 2, shape: float = 4, lamda: float = 0, seed: Optional[int] = None) -> Array:
    """
    (Standardized Generalized Hyperbolic) Random Number Function (rsgh)

    :param n: the number of draws
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lamda: the second shape parameter of the GH distribution related to the GIG distribution.
    :param seed: an optional value to initialize the random seed generator 
    :rtype: a numpy array
    """
    parameters = np.asarray([mu, sigma, skew, shape, lamda], dtype = np.float64)
    params = paramgh(parameters[2], parameters[3], parameters[4])
    params = paramghconvert(params)
    r = genhyperbolic.rvs(size = n, a = params[0], b = params[1], loc = params[2], scale = params[3], p = params[4], random_state = seed)
    r = parameters[0] + r * parameters[1]
    return np.atleast_1d(r)

# Generalized Hyperbolic Skew Student Distribution
# Much of the code translated from David Scott's SkewHyperbolic R package
# for the non standardized case

def dghst(x: Vector, params: Vector, log=False, tolerance=np.finfo(float).eps**0.5) -> Array:
    """
    Non Standardized Skew Hyperbolic distribution.
    # params [mu, delta, beta, nu]
    """
    x = np.atleast_1d(x)
    params = np.asarray(params, dtype=np.float64)
    beta_squared = np.power(params[2], 2.0)
    delta_squared = np.power(params[1], 2.0)
    res_squared = np.power(x - params[0], 2.0)
    res = x - params[0]
    
    if np.abs(params[2]) > tolerance:
        m_value = np.sqrt(np.power(params[1], 2.0) + res_squared)        
        if params[2] < tolerance:
            m = kve((params[3] + 1.0) / 2.0, -1.0 * params[2] * m_value)
            d = (np.power(2.0, ((1 - params[3]) / 2.0)) * np.power(params[1], params[3]) * 
                 np.power(np.abs(params[2]), ((params[3] + 1.0) / 2.0)) * m * 
                 np.exp(params[2] * (res + m_value)) / 
                 (gamma(params[3] / 2.0) * np.sqrt(np.pi) * np.power(m_value, ((params[3] + 1.0) / 2.0))))
        else:
            m = kve((params[3] + 1.0) / 2.0, params[2] * m_value)
            d = (np.power(2.0, ((1.0 - params[3]) / 2.0)) * np.power(params[1], params[3]) * 
                 np.power(np.abs(params[2]), ((params[3] + 1.0) / 2.0)) * m * np.exp(params[2] * (res - m_value)) / 
                 (gamma(params[3] / 2.0) * np.sqrt(np.pi) * np.power(m_value, ((params[3] + 1.0) / 2.0))))
    else:
        d = (gamma((params[3] + 1.0) / 2.0) / (np.sqrt(np.pi) * params[1] * gamma(params[3] / 2.0)) * 
             np.power(1.0 + (res_squared) / delta_squared, (-(params[3] + 1.0) / 2.0)))
    
    if log:
        d = np.log(d)
    
    return d

def skewhyp_steps_size(dist: Vector, params: Vector, side: str = 'left')-> Array:
    params = np.asarray(params, dtype=np.float64)
    step = params[1]
    if params[2] > 0:
        if side == 'right':
            step = params[1] * np.abs(params[2]) * np.power(params[3] * dist, -2.0/params[3])
    if params[2] < 0:
        if side == 'left':
            step = params[1] * np.abs(params[2]) * np.power(params[3] * dist, -2.0/params[3])
    if params[2] == 0.0:
        step = np.exp(np.divide(dist,params[3]))
    
    return step

def skewhyp_calc_range(params: Vector, density: bool = True, tol: float = 1e-5)-> Array:
    params = np.asarray(params, dtype=np.float64)
    mode = ghstmode(params)
    if density == True:
        x_high = mode + params[1]
        while (dghst(x_high, params)[0] > tol):            
            x_high = x_high + skewhyp_steps_size(dghst(x_high, params) - tol, params, side = 'right')
        
        x_low = mode - params[1]
        while (dghst(x_low, params) > tol):
            x_low = x_low - skewhyp_steps_size(dghst(x_low, params) - tol, params, side = 'left')
        
        zero_fun = lambda x: dghst(x, params)[0] - tol
        x_upper = brentq(zero_fun, a = mode[0], b = x_high[0])
        x_lower = brentq(zero_fun, a = x_low[0], b = mode[0])
    else:
        ghst_int = lambda x: dghst(x, params)[0]
        upper_prob = lambda x: quad(ghst_int, a = x, b = np.Inf)[0]
        x_high = mode + params[1]
        while upper_prob(x_high[0]) > tol :
            x_high = x_high + skewhyp_steps_size(upper_prob(x_high[0]) - tol, params, side = 'right')
            
        lower_prob = lambda x: quad(ghst_int, a = -np.Inf, b = x)[0]
        x_low = mode - params[1]
        while  lower_prob(x_low[0]) > tol :
            x_low = x_low - skewhyp_steps_size(lower_prob(x_low[0]) - tol, params, side = 'left')
        
        zero_fun = lambda x: upper_prob(x) - tol
        x_upper = brentq(zero_fun, a = mode[0], b = x_high[0])
        zero_fun = lambda x: lower_prob(x) - tol
        x_lower = brentq(zero_fun, a = x_low[0], b = mode[0])
    
    range = np.asarray([x_lower, x_upper], dtype=np.float64)
    return range

def ghstmode(params: Vector) -> Array:
    params = np.asarray(params, dtype=np.float64)
    mode_fun = lambda x: -1.0 * dghst(x, params, log = True)
    opt = minimize(mode_fun, x0 = params[0], method = "BFGS")
    if opt.success == True:
        return opt.x
    else:
        return params[0]

def pghst(q: Vector, params: Vector, lower_tail:bool = True):
    mode = ghstmode(params)
    q = np.atleast_1d(q)
    q = q.astype(np.float64)
    finite_condition = np.isfinite(q)
    less_than_condition = np.logical_and(q <= mode, finite_condition)
    greater_than_condition = np.logical_and(q > mode, finite_condition)
    p = np.zeros(shape = np.size(q), dtype=np.float64)
    lower = np.where(less_than_condition)[0]
    upper = np.where(greater_than_condition)[0]
    fun = lambda x: dghst(x, params)[0]
    if np.size(lower) > 0:
        for i in lower:
            p[i] = quad(fun, a = -np.Inf, b = q[i], limit = 100)[0]
    if np.size(upper) > 0:
        for i in upper:
            p[i] = quad(fun, a = q[i], b = np.Inf, limit = 100)[0]
    if lower_tail:
        if np.size(upper) > 0:
            p[upper] = 1- p[upper]
    else:
        if np.size(lower) > 0:
            p[lower] = 1- p[lower]
    
    return p

def qghst(p: Vector, params: Vector, lower_tail:bool = True):
    p = np.atleast_1d(p)
    p = p.astype(np.float64)
    if lower_tail == False:
        p = 1.0 - p
    params = np.asarray(params, dtype=np.float64)
    mode = ghstmode(params)
    p_mode =  pghst(mode, params)
    x_range = skewhyp_calc_range(params, tol = 10**(-7))
    less = np.where((p <= p_mode) & (p > 0))[0]
    q = np.zeros(shape = np.size(p), dtype=np.float64)
    if len(less) > 0:
        p_low = np.min(p[less])
        x_low = mode - skewhyp_steps_size(params[1], params, side = 'left')
        
        while pghst(x_low, params) >= p_low:
             x_low -= skewhyp_steps_size(mode - x_low, params, side = 'left')
        
        x_range = [x_low, mode[0]]
        fn = lambda x,p: pghst(x, params)[0] - p
        for i in less:
            q[i] = brentq(fn, a = x_range[0], b = x_range[1], args = p[i])
    
    greater = np.where((p > p_mode) & (p < 1))[0]
    p[greater] = 1.0 - p[greater]
    if len(greater) > 0:
        p_high = np.min(p[greater])
        x_high = mode + skewhyp_steps_size(params[1], params, side = 'right')
        while pghst(x_high, params, lower_tail=False) >= p_high:
            x_high += skewhyp_steps_size(x_high - mode, params, side = 'right')
        
        x_range = [mode[0], x_high[0]]

        fn = lambda x,p: pghst(x, params, lower_tail = False)[0] - p
        for i in greater:
            q[i] = brentq(fn, a = x_range[0], b = x_range[1], args = p[i])
    
    return q

# Standardized Generalized Hyperbolic Skew Student Distribution

def dsghst(x: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, log: bool = False) -> Array:
    """
    (Standardized Generalized Hyperbolic Skew Student) Probability Density Function (dsghst)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to x 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param x: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param log: whether to return the log density
    :rtype: a numpy array
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(dsghst)
        return v_fun(x, mu = mu, sigma = sigma, skew = skew, shape = shape, log = log)

    parameters = np.array([mu, sigma, skew, shape], dtype=np.float64)
    x = scaler(x, parameters[0], parameters[1])
    params = paramghst(parameters[2],parameters[3])
    # [mu, delta, beta, nu]
    beta_squared = np.power(params[2], 2.0)
    delta_squared = np.power(params[1], 2.0)
    res_squared = np.power(x - params[0], 2.0)
    pdf = ((1.0 - params[3])/2.0) * np.log(2.0) + params[3] * np.log(params[1]) + ((params[3] + 1.0)/2.0)\
          * np.log(np.abs(params[2])) + np.log(kve((params[3] + 1.0)/2.0,  np.sqrt(beta_squared * (delta_squared + res_squared))))\
          - np.sqrt(beta_squared * (delta_squared + res_squared)) + params[2] * (x - params[0]) - gammaln(params[3]/2.0) - np.log(np.pi)/2.0\
              - ((params[3] + 1.0)/2.0) * np.log(delta_squared + res_squared)/2.0
    pdf = np.exp(pdf)/parameters[1]
    if log == True:
        pdf = np.log(pdf)
    return np.atleast_1d(pdf)

def psghst(q: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lower_tail: bool = True) -> Array:
    """
    (Standardized Generalized Hyperbolic Skew Student) Cumulative Probability Function (psghst)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to q 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param q: a vector of quantiles
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    q = np.atleast_1d(q)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(psghst)
        return v_fun(q, mu = mu, sigma = sigma, skew = skew, shape = shape, lower_tail = lower_tail)

    params = paramghst(skew, shape)
    # [mu, delta, beta, nu]
    params[0] = params[0] * sigma + mu
    params[1] = params[1] * sigma
    params[2] = params[2] / sigma
    p = pghst(q, params, lower_tail=lower_tail)
    return np.atleast_1d(p)

def qsghst(p: Vector, mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 4, lower_tail: bool = True) -> Array:
    """
    (Standardized Generalized Hyperbolic Skew Student) Quantile Function (qsghst)

    In cases when the parameters are vectors of size n (>1), then they must be equal in size to p 
    (unless they are of size 1 in which case they are recycled). This is achieved using the numpy 
    `vectorize` function.

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
    :rtype: a numpy array
    """
    p = np.atleast_1d(p)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)

    if any(param.size > 1 for param in [mu, sigma, skew, shape]):
        v_fun = np.vectorize(qsghst)
        return v_fun(p, mu = mu, sigma = sigma, skew = skew, shape = shape, lower_tail = lower_tail)

    params = paramghst(skew, shape)
    params[0] = params[0] * sigma + mu
    params[1] = params[1] * sigma
    params[2] = params[2] / sigma
    q = qghst(p, params, lower_tail = lower_tail)    
    return np.atleast_1d(q)

def rsghst(n: int = 1, mu: float = 0, sigma: float = 1, skew: float = 2, shape: float = 4, seed: Optional[int] = None) -> Array:
    """
    (Standardized Generalized Hyperbolic Skew Student) Random Number Function (rsghst)

    :param p: a vector of probabilities
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param seed: an optional value to initialize the random seed generator 
    :rtype: a numpy array
    """
    parameters = np.asarray([mu, sigma, skew, shape], dtype=np.float64)
    params = paramghst(skew, shape)
    # [mu, delta, beta, nu]
    rgam = 1.0/gammadist.rvs(a = params[3]/2.0, scale = 2.0/np.power(params[1], 2.0), loc = 0.0, size = n, random_state = seed)
    sig = np.sqrt(rgam)
    z = norm.rvs(size = n, loc = 0, scale = 1, random_state = seed)
    r = params[0] + params[2] * np.power(sig, 2.0) + sig * z
    r = parameters[0] + r * parameters[1]
    return np.atleast_1d(r)

def skewness_norm(mu: Vector = 0, sigma: Vector = 1)->Vector:
    mu = np.atleast_1d(mu)
    skewness = np.atleast_1d([0] * len(mu))
    return skewness

def skewness_std(mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5)->Vector:
    shape = np.atleast_1d(shape)
    skewness = np.atleast_1d([0] * len(shape))
    return skewness

def skewness_ged(mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5)->Vector:
    shape = np.atleast_1d(shape)
    skewness = np.atleast_1d([0] * len(shape))
    return skewness

def skewness_snorm(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 5)->Vector:
    skew = np.atleast_1d(skew)
    sigma = np.atleast_1d(sigma)
    m1 = 2.0/np.sqrt(2.0 * np.pi)
    skew_squared = skew ** 2
    m3 = 4.0/np.sqrt(2 * np.pi)
    skewness = (skew - 1/skew) * ((m3 + 2 * m1 ** 3 - 3.0 * m1) * (skew_squared + (1/skew_squared)) + 3.0 * m1 - 4.0 * m1 ** 3) / (((1.0 - m1 ** 2) * (skew_squared + 1/skew_squared) + 2 * m1 **2 - 1.0)**(3/2))
    return skewness

def skewness_sstd(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 5, shape:Vector = 5)->Vector:
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    eta = shape
    k2 = skew ** 2
    lda = (k2 - 1.0) / (k2 + 1.0)
    lda_squared = lda ** 2
    eta_minus_2 = eta - 2.0
    eta_minus_1 = eta - 1.0
    eta_minus_3 = eta - 3.0
    eta_minus_4 = eta - 4.0
    ep1 = (eta + 1.0) / 2.0
    lnc = gammaln(ep1) - gammaln(eta / 2.0) - 0.5 * np.log(np.pi * eta_minus_2)
    cx = np.exp(lnc)
    a = 4.0 * lda * cx * eta_minus_2 / eta_minus_1
    a_squared = a ** 2
    b = np.sqrt(1.0 + 3.0 * lda_squared - a_squared)
    my2 = 1.0 + 3.0 * lda_squared
    my3 = 16 * cx * lda * (1.0 + lda_squared) * (eta_minus_2 ** 2) / (eta_minus_1 * eta_minus_3)
    my4 = 3.0 * eta_minus_2 * (1.0 + 10.0 * lda_squared + 5.0 * lda_squared * lda_squared) / eta_minus_4
    b_cubed = b ** 3
    # Calculate skewness
    skewness = (my3 - 3 * a * my2 + 2 * a_squared * a) / b_cubed
    return skewness

def skewness_sged(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 5, shape:Vector = 5)->Vector:
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    lambda_ = np.sqrt(2 ** (-2 / shape) * gamma(1 / shape) / gamma(3 / shape))
    m1 = ((2 ** (1 / shape) * lambda_) ** 1 * gamma(2 / shape) / gamma(1 / shape))
    m2 = 1  # This is a constant in your R code
    m3 = ((2 ** (1 / shape) * lambda_) ** 3 * gamma(4 / shape) / gamma(1 / shape))
    skew_squared = skew ** 2
    inv_skew = 1 / skew
    inv_skew_squared = inv_skew ** 2
    m1_cubed = m1 ** 3
    numerator = (skew - inv_skew) * ((m3 + 2 * m1_cubed - 3 * m1 * m2) * (skew_squared + inv_skew_squared) + 3 * m1 * m2 - 4 * m1_cubed)
    denominator = ((m2 - m1 ** 2) * (skew_squared + inv_skew_squared) + 2 * m1 ** 2 - m2) ** (3 / 2)
    if denominator == 0:
        raise ValueError("The denominator of the skewness calculation cannot be zero.")
    skewness = numerator / denominator
    return skewness

def skewness_jsu(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 5, shape:Vector = 5)->Vector:
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    omega = -skew / shape
    w = np.exp((1.0 * shape) ** -2)
    w_minus_1 = w - 1
    w_plus_2 = w + 2
    s3 = -0.25 * np.sqrt(w) * (w_minus_1**2) * (w * w_plus_2 * np.sinh(3 * omega) + 3 * np.sinh(omega))
    # Calculate denominator
    denominator = 0.5 * w_minus_1 * (w * np.cosh(2 * omega) + 1)
    if np.any(denominator == 0):
        raise ValueError("The denominator of the skewness calculation cannot be zero.")
    skewness = s3 / (denominator ** (3 / 2))
    return skewness

def skewness_sghst(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 5, shape:Vector = 7)->Vector:
    params = paramghst(betabar=skew, nu=shape)
    delta = params[1]
    beta = params[2]
    nu = params[3]
    beta2 = beta * beta
    delta2 = delta * delta
    skewness = ((2 * np.sqrt(nu - 4) * beta * delta) / ((2 * beta2 * delta2 + (nu - 2) * (nu - 4)) ** (3/2))) * \
                (3 * (nu - 2) + ((8 * beta2 * delta2) / (nu - 6)))
    skewness = np.atleast_1d(skewness)
    return skewness


def skewness_sgh(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 0.9, shape:Vector = 7, lamda:Vector = -0.5)->Vector:
    parameters = np.asarray([mu, sigma, skew, shape, lamda], dtype=np.float64)
    params = paramgh(parameters[2], parameters[3], parameters[4])
    params = paramghconvert(params)
    skewness = genhyperbolic.stats(p = params[4], a = params[0], b = params[1], loc = params[2], scale = params[3], moments = 's')
    skewness = np.atleast_1d(skewness)
    return skewness

def kurtosis_norm(mu: Vector = 0, sigma: Vector = 1)->Vector:
    mu = np.atleast_1d(mu)
    kurtosis = np.atleast_1d(3.0 * len(mu))
    return kurtosis

def kurtosis_std(mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5)->Vector:
    shape = np.atleast_1d(shape)
    kurtosis = np.full_like(shape, np.nan, dtype=np.float64)
    # Indices where shape > 4
    valid_indices = shape > 4
    if np.any(valid_indices):
        shape_valid = shape[valid_indices]
        denominator = (shape_valid - 4.0)
        kurtosis[valid_indices] = (6.0 / denominator) + 3
    return kurtosis

def kurtosis_ged(mu: Vector = 0, sigma: Vector = 1, shape: Vector = 5)->Vector:
    shape = np.atleast_1d(shape)
    kurtosis = (((gamma(1.0/shape) / gamma(3.0/shape)) ** 2) * (gamma(5.0/shape) / gamma(1.0/shape)))
    return kurtosis

def kurtosis_snorm(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 5)->Vector:
    skew = np.atleast_1d(skew)
    kurtosis = np.atleast_1d([3.0] * len(skew))
    return kurtosis

def kurtosis_sstd(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 5)->Vector:
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    kurtosis = np.full_like(shape, np.nan, dtype=np.float64)
    valid_indices = shape > 4
    if np.any(valid_indices):
        skew_valid = skew[valid_indices]
        shape_valid = shape[valid_indices]
        eta = shape_valid
        k2 = skew_valid ** 2
        lda = (k2 - 1) / (k2 + 1)
        lda_squared = lda ** 2
        lda_fourth = lda_squared ** 2
        ep1 = (eta + 1) / 2
        lnc = gammaln(ep1) - gammaln(eta / 2) - 0.5 * np.log(np.pi * (eta - 2))
        cx = np.exp(lnc)
        a = 4 * lda * cx * (eta - 2) / (eta - 1)
        a_squared = a ** 2
        a_fourth = a ** 4
        b = np.sqrt(1 + 3 * lda_squared - a_squared)
        b_squared = 1 + 3 * lda_squared - a_squared
        b_fourth = b_squared ** 2
        my2 = 1 + 3 * lda_squared
        my3 = 16 * cx * lda * (1 + lda_squared) * ((eta - 2) ** 2) / ((eta - 1) * (eta - 3))
        my4 = 3 * (eta - 2) * (1 + 10 * lda_squared + 5 * lda_fourth) / (eta - 4)
        m4_valid = -3 + (my4 - 4 * a * my3 + 6 * a_squared * my2 - 3 * a_fourth) / b_fourth
        kurtosis[valid_indices] = m4_valid + 3.0

    return kurtosis

def kurtosis_sged(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 5)->Vector:
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    lambda_ = np.sqrt(2**(-2/shape) * gamma(1/shape) / gamma(3/shape))
    m1 = (2**(1/shape) * lambda_) * gamma(2/shape) / gamma(1/shape)
    m2 = 1
    m3 = ((2**(1/shape) * lambda_)**3) * gamma(4/shape) / gamma(1/shape)
    m4 = ((2**(1/shape) * lambda_)**4) * gamma(5/shape) / gamma(1/shape)
    cm4 = (-3 * m1**4 * (skew - 1/skew)**4) + \
          (6 * m1**2 * (skew - 1/skew)**2 * m2 * (skew**3 + 1/skew**3)) / (skew + 1/skew) - \
          (4 * m1 * (skew - 1/skew) * m3 * (skew**4 - 1/skew**4)) / (skew + 1/skew) + \
          (m4 * (skew**5 + 1/skew**5)) / (skew + 1/skew)
    denominator = ((m2 - m1**2) * (skew**2 + 1/skew**2) + 2 * m1**2 - m2) ** 2
    invalid_indices = denominator <= 0
    denominator[invalid_indices] = np.nan
    kurtosis = (cm4 / denominator)
    return kurtosis


def kurtosis_jsu(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 2, shape: Vector = 5)->Vector:
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    omega = -skew / shape
    w = np.exp(shape**-2.0)
    s4 = 0.125 * (w - 1)**2 * (w**2 * (w**4 + 2 * w**3 + 3 * w**2 - 3) * np.cosh(4 * omega) +
                               4 * w**2 * (w + 2) * np.cosh(2 * omega) + 3 * (2 * w + 1))
    kurtosis = s4 / (0.5 * (w - 1) * (w * np.cosh(2 * omega) + 1))**2
    return kurtosis

def kurtosis_sghst(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 0, shape: Vector = 5)->Vector:
    skew = np.atleast_1d(skew)
    shape = np.atleast_1d(shape)
    params = paramghst(betabar=skew, nu=shape)
    delta = params[1]
    beta = params[2]
    nu = params[3]
    beta2 = beta * beta
    delta2 = delta * delta
    k1 = 6 / ((2 * beta2 * delta2 + (nu - 2) * (nu - 4)) ** 2)
    k21 = (nu - 2) * (nu - 2) * (nu - 4)
    k22 = (16 * beta2 * delta2 * (nu - 2) * (nu - 4)) / (nu - 6)
    k23 = (8 * (beta2 ** 2) * (delta2 ** 2) * (5 * nu - 22)) / ((nu - 6) * (nu - 8))
    kurtosis = k1 * (k21 + k22 + k23) + 3.0
    return kurtosis

def kurtosis_sgh(mu: Vector = 0, sigma: Vector = 1, skew: Vector = 0.9, shape:Vector = 7, lamda:Vector = -0.5)->Vector:
    parameters = np.asarray([mu, sigma, skew, shape, lamda], dtype=np.float64)
    params = paramgh(parameters[2], parameters[3], parameters[4])
    params = paramghconvert(params)
    kurtosis = np.add(genhyperbolic.stats(p = params[4], a = params[0], b = params[1], loc = params[2], scale = params[3], moments = 'k'),3.0)
    kurtosis = np.atleast_1d(kurtosis)
    return kurtosis

def kurtosis(distribution:str = 'std', mu: Vector = 0, sigma: Vector = 1, skew: Vector = 0.9, shape:Vector = 7, lamda:Vector = -0.5)->Array:
    valid_distribution = _validate_distribution(distribution)
    if not valid_distribution:
        raise ValueError("The distribution is not valid.")
    fun = globals()[f'kurtosis_{distribution}']
    params = inspect.signature(fun).parameters
    args = {}
    if 'mu' in params:
        args['mu'] = mu
    if 'sigma' in params:
        args['sigma'] = sigma
    if 'skew' in params:
        args['skew'] = skew
    if 'shape' in params:
        args['shape'] = shape
    if 'lamda' in params:
        args['lamda'] = lamda
    return fun(**args)

def skewness(distribution:str = 'std', mu: Vector = 0, sigma: Vector = 1, skew: Vector = 0.9, shape:Vector = 7, lamda:Vector = -0.5)->Array:
    valid_distribution = _validate_distribution(distribution)
    if not valid_distribution:
        raise ValueError("The distribution is not valid.")
    fun = globals()[f'skewness_{distribution}']
    params = inspect.signature(fun).parameters
    args = {}
    if 'mu' in params:
        args['mu'] = mu
    if 'sigma' in params:
        args['sigma'] = sigma
    if 'skew' in params:
        args['skew'] = skew
    if 'shape' in params:
        args['shape'] = shape
    if 'lamda' in params:
        args['lamda'] = lamda
    return fun(**args)