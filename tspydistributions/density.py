import torch
from scipy.special import kve
from jaxtyping import Float

Tensor = Float[torch.Tensor, "..."]

def _scale_fun(x, mu, sigma):
    return (x - mu)/sigma


def torch_kve(nu, z):
    return torch.tensor(kve(nu,z))

def betaln(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b)

def _kappagh(x, lamda):
    if lamda ==  -0.5:
        kappa = torch.divide(torch.tensor(1.0),x)
    else:
        kappa = (torch_kve(torch.add(lamda,1.0), x)/torch_kve(lamda, x))/x
    return kappa

def _deltakappagh(x, lamda):
    if lamda == -0.5:
        deltakappa = torch.subtract(_kappagh(x, torch.add(lamda, 1.0)), _kappagh(x, lamda))
    else:
        deltakappa = torch.subtract(_kappagh(x, torch.add(lamda,1.0)), _kappagh(x, lamda))
    return deltakappa
 
def _paramgh(rho, zeta, lamda):
    rho2 = 1 - torch.pow(rho, 2.0)
    alpha = torch.pow(zeta, 2.0) * _kappagh(zeta, lamda)/rho2
    alpha = torch.sqrt(alpha * (1.0 + torch.multiply(torch.pow(rho, 2.0), torch.pow(zeta, 2.0)) * _deltakappagh(zeta, lamda)/rho2))
    beta = alpha * rho
    delta = torch.divide(zeta, (alpha * torch.sqrt(rho2)))
    mu = torch.multiply(torch.multiply(torch.tensor(-1.0), beta), torch.pow(delta, 2.0)) * _kappagh(zeta, lamda)
    params = torch.tensor([alpha, beta, delta, mu, lamda])
    return params

def _paramghst(betabar, nu):
    delta = torch.sqrt(1.0/(((2.0 * torch.pow(betabar,2.0))/(torch.pow(torch.subtract(nu,2.0), 2.0) * torch.subtract(nu,4.0))) + (1.0/torch.subtract(nu,2.0))))
    beta = betabar / delta
    mu = -1.0 * ((beta * torch.pow(delta, 2.0)) / torch.subtract(nu,2.0))
    params =  torch.tensor([mu, delta, beta, nu])
    return params

def _fs_skew_moments(m, skew):
    m_squared = torch.pow(m, 2.0)
    skew_squared = torch.pow(skew, 2.0)
    mu = m * torch.subtract(skew, torch.divide(torch.tensor(1.0),skew))
    sigma = torch.sqrt((1. - m_squared) * (skew_squared + 1.0/skew_squared) + 2.0 * m_squared - 1.0)
    out = [mu, sigma]
    return out


def djsu(parameters:Tensor, x:Tensor, log:bool = False)->Tensor:
    """
    (Johnson's SU) Probability Density Function (djsu)

    This is a torch implementation of the probability density function of the Johnson's SU distribution
    useful for inclusion with PyTorch models.

    :param parameters: a tensor of parameters [mu, sigma, skew, shape].
    :param x: a tensor of quantiles
    :param log: whether to return the log density
    :rtype: a torch tensor
    """

    x = _scale_fun(x, parameters[0], parameters[1])
    rtau = 1.0/parameters[3]
    if torch.lt(rtau, torch.tensor(1e-7)):
        w = torch.tensor(1.0)
    else:
        w = torch.exp(torch.pow(rtau, 2.0))
    omega = -1.0 * parameters[2] * rtau
    c = torch.sqrt(1.0/(0.5 * (w - 1.0) * (w * torch.cosh(2.0 * omega) + 1.0)))
    z = (x - (c * torch.sqrt(w) * torch.sinh(omega)))/c
    r = - 1.0 * parameters[2] + torch.arcsinh(z)/rtau
    pdf = -1.0 * torch.log(c) - torch.log(rtau) - 0.5 * torch.log(torch.pow(z,2.0) + 1.0) - 0.5 * torch.log(torch.tensor(torch.pi * 2.0)) - 0.5 * torch.pow(r, 2.0)
    pdf = pdf - torch.log(parameters[1])
    if log:
        return pdf
    else:
        return torch.exp(pdf)
    
def dnorm(parameters:Tensor, x:Tensor, log:bool = False)->Tensor:
    """
    (Normal) Probability Density Function (dnorm)

    This is a torch implementation of the probability density function of the Normal distribution
    useful for inclusion with PyTorch models.

    :param parameters: a tensor of parameters [mu, sigma].
    :param x: a tensor of quantiles
    :param log: whether to return the log density
    :rtype: a torch tensor
    """

    pdf = torch.distributions.normal.Normal(parameters[0],parameters[1]).log_prob(x)
    if log:
        return pdf
    else:
        return torch.exp(pdf)

    
def dstd(parameters:Tensor, x:Tensor, log:bool = False)->Tensor:
    """
    (Student's t) Probability Density Function (dstd)
    
    This is a torch implementation of the probability density function of the Student's t distribution
    useful for inclusion with PyTorch models.

    :param parameters: a tensor of parameters [mu, sigma, shape].
    :param x: a tensor of quantiles
    :param log: whether to return the log density
    :rtype: a torch tensor
    """
    
    x = _scale_fun(x, parameters[0], parameters[1])
    scale = 0.5 * (torch.log(parameters[2]) - torch.log(parameters[2] - 2.0))
    x = x * scale.exp()
    alpha = torch.lgamma((parameters[2] + 1.0)/2.0) - 0.5 * (torch.log(torch.tensor(torch.pi)) + torch.log(parameters[2]))
    beta = torch.lgamma(parameters[2]/2.0) + 0.5 * (parameters[2] + 1.0) * torch.log((1.0 + torch.square(x)/parameters[2]))
    ratio = alpha - beta
    pdf = scale + ratio - torch.log(parameters[1])
    if log:
        return pdf
    else:
        return torch.exp(pdf)

def dged(parameters:Tensor, x:Tensor, log:bool = False)->Tensor:
    """
    (GED) Probability Density Function (dged)

    This is a torch implementation of the probability density function of the Generalized Error Distribution
    useful for inclusion with PyTorch models.

    :param parameters: a tensor of parameters [mu, sigma, shape].
    :param x: a tensor of quantiles
    :param log: whether to return the log density
    :rtype: a torch tensor
    """

    x = _scale_fun(x, parameters[0], parameters[1])
    lam = 0.5 * ((2./parameters[2]) * torch.log(torch.tensor(0.5)) + torch.lgamma(1.0/parameters[2]) - torch.lgamma(3.0/parameters[2]))
    g = torch.log(parameters[2]) - (lam + (1. + (1./parameters[2])) * torch.log(torch.tensor(2.)) + torch.lgamma(1/parameters[2]))
    pdf = g + (-0.5 * torch.pow(torch.abs(x/lam.exp()), parameters[2]))
    pdf = pdf - torch.log(parameters[1])
    if log:
        return pdf
    else:
        return torch.exp(pdf)


def dsnorm(parameters:Tensor, x:Tensor, log:bool = False)->Tensor:
    """
    (Skew-Normal) Probability Density Function (dsnorm)

    This is a torch implementation of the probability density function of the Skew-Normal distribution
    useful for inclusion with PyTorch models.

    :param parameters: a tensor of parameters [mu, sigma, skew].
    :param x: a tensor of quantiles
    :param log: whether to return the log density
    :rtype: a torch tensor
    """

    x = _scale_fun(x, parameters[0], parameters[1])
    m = torch.divide(torch.tensor(2.0),torch.sqrt(torch.tensor(2.0 * torch.pi)))
    fs_mu, fs_sigma = _fs_skew_moments(m, parameters[2])
    z = x * fs_sigma + fs_mu
    z_sign = torch.sign(z)
    xi = torch.pow(parameters[2], z_sign)
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    pdf = torch.log(g) + dnorm(parameters = torch.tensor([0., 1.]), x = z/xi, log = True) + torch.log(fs_sigma)
    pdf = pdf - torch.log(parameters[1])
    if log:
        return pdf
    else:
        return torch.exp(pdf)
    
def dsged(parameters:Tensor, x:Tensor, log:bool = False)->Tensor:
    x = _scale_fun(x, parameters[0], parameters[1])
    lam = torch.sqrt(torch.pow(2.0,-2.0/parameters[3]) * torch.lgamma(1.0/parameters[3]).exp()/torch.lgamma(3.0/parameters[3]).exp())
    m = torch.pow(2.0, (1.0/parameters[3])) * lam * torch.lgamma(2.0/parameters[3]).exp()/torch.lgamma(1.0/parameters[3]).exp()
    fs_mu, fs_sigma = _fs_skew_moments(m, parameters[2])
    z = x * fs_sigma + fs_mu
    z_sign = torch.sign(z)
    xi = torch.pow(parameters[2], z_sign)
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    new_pars = torch.cat([torch.tensor([0., 1.]), parameters[3].unsqueeze(0)])
    ged_log_lik = dged(parameters = new_pars, x = z/xi, log = True)
    pdf = torch.log(g) + (torch.tensor(-1.0) * ged_log_lik) + torch.log(fs_sigma)
    pdf = pdf - torch.log(parameters[1])
    if log:
        return pdf
    else:
        return torch.exp(pdf)
    

def dsstd(parameters:Tensor, x:Tensor, log:bool = False)->Tensor:
    """
    (Skew-Student's t) Probability Density Function (dsstd)

    This is a torch implementation of the probability density function of the Skew-Student's t distribution
    useful for inclusion with PyTorch models.

    :param parameters: a tensor of parameters [mu, sigma, skew, shape].
    :param x: a tensor of quantiles
    :param log: whether to return the log density
    :rtype: a torch tensor
    """

    x = _scale_fun(x, parameters[0], parameters[1])
    m = 2.0 * torch.sqrt(parameters[3] - 2.0)/(parameters[3] - 1.0)/torch.exp(betaln(torch.tensor(0.5), parameters[3]/2.0))
    fs_mu, fs_sigma = _fs_skew_moments(m, parameters[2])
    z = x * fs_sigma + fs_mu
    z_sign = torch.sign(z)
    xi = torch.pow(parameters[2], z_sign)
    g = 2.0/(parameters[2] + 1.0/parameters[2])
    new_pars = torch.cat([torch.tensor([0., 1.]), parameters[3].unsqueeze(0)])
    std_log_lik = dstd(parameters = new_pars, x = z/xi, log = True)
    pdf = torch.log(g) + (torch.tensor(-1.0) * std_log_lik) + torch.log(fs_sigma)
    pdf = pdf - torch.log(parameters[1])
    if log:
        return pdf
    else:
        return torch.exp(pdf)

def dsgh(parameters:Tensor, x:Tensor, log:bool = False)->Tensor:
    """
    (Standardized Generalized Hyperbolic) Probability Density Function (dsgh)

    This is a torch implementation of the probability density function of the Standardized Generalized Hyperbolic distribution
    useful for inclusion with PyTorch models. Currently due to the absence of a torch implementation of the modified Bessel 
    function of the second kind, the pdqr function should be used instead.

    :param parameters: a tensor of parameters [mu, sigma, skew, shape, lamda].
    :param x: a tensor of quantiles
    :param log: whether to return the log density
    :rtype: a torch tensor
    """

    x = _scale_fun(x, parameters[0], parameters[1])
    params = _paramgh(parameters[2],parameters[3], parameters[4])
    arg1 = params[2] * torch.sqrt(torch.pow(params[0], 2.0) - torch.pow(params[1], 2.0))
    a = (params[4]/2.0) * torch.log(torch.pow(params[0],2.0) - torch.pow(params[1],2.0))\
          - (torch.log(torch.sqrt(torch.tensor(2.0 * torch.pi))) + (params[4] - 0.5) * torch.log(params[0])\
             + params[4] * torch.log(params[2]) + torch.log(torch_kve(params[4], arg1)) - arg1)
    f = ((params[4] - 0.5)/2.0) * torch.log(torch.pow(params[2],2.0) + torch.pow((x - params[3]),2.0))
    arg2 = params[0] * torch.sqrt(torch.pow(params[2],2.0) + torch.pow((x - params[3]), 2.0))
    k = torch.log(torch_kve(params[4] - 0.5, arg2)) - arg2
    e = params[1] * (x - params[3])
    pdf = a + f + k + e
    pdf = pdf - torch.log(parameters[1])
    if log:
        return pdf
    else:
        return torch.exp(pdf)

def dsghst(parameters:Tensor, x:Tensor, log:bool = False)->Tensor:
    """
    (Standardized Generalized Hyperbolic Skew Student) Probability Density Function (dsghst)

    This is a torch implementation of the probability density function of the Standardized Generalized Hyperbolic Skew Student distribution
    useful for inclusion with PyTorch models. Currently due to the absence of a torch implementation of the modified Bessel function of 
    the second kind, the pdqr function should be used instead.

    :param parameters: a tensor of parameters [mu, sigma, skew, shape].
    :param x: a tensor of quantiles
    :param log: whether to return the log density
    :rtype: a torch tensor
    """

    x = _scale_fun(x, parameters[0], parameters[1])
    params = _paramghst(parameters[2],parameters[3])
    beta_squared = torch.pow(params[2], 2.0)
    delta_squared = torch.pow(params[1], 2.0)
    res_squared = torch.pow(x - params[0], 2.0)
    pdf = ((1.0 - params[3])/2.0) * torch.log(torch.tensor(2.0)) + params[3] * torch.log(params[1]) + ((params[3] + 1.0)/2.0)\
          * torch.log(torch.abs(params[2])) + torch.log(torch_kve((params[3] + 1.0)/2.0,  torch.sqrt(beta_squared * (delta_squared + res_squared))))\
          - torch.sqrt(beta_squared * (delta_squared + res_squared)) + params[2] * (x - params[0]) - torch.lgamma(params[3]/2.0) - torch.log(torch.tensor(torch.pi))/2.0\
              - ((params[3] + 1.0)/2.0) * torch.log(delta_squared + res_squared)/2.0
    pdf = pdf - torch.log(parameters[1])
    if log:
        return pdf
    else:
        return torch.exp(pdf)
    
  
