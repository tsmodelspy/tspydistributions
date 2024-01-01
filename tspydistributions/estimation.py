import torch
from tspydistributions.helpers import _distribution_bounds, _default_parameter_values, _validate_fixed, _sublist
import numpy as np
from typing import List, Dict, Literal, Any, Optional, TYPE_CHECKING, TypeVar, Callable
import numpy.typing as npt
from scipy.optimize import minimize, Bounds
from scipy.special import kve
import numdifftools as nd
from tspydistributions.numderiv import get_hessian, get_jacobian

Array = npt.NDArray[np.float64]
Vector = npt.ArrayLike

def torch_kve(nu, z):
    return torch.tensor(kve(nu,z))

def dnorm(x):
    return torch.distributions.normal.Normal(0.,1.).log_prob(x)

def betaln(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b)

def _fdestimation(dist_name: str, x: Vector, fixed: Optional[Dict[str,float]] = None, method:str = 'L-BFGS-B', tol:float = 1e-08, options:dict = {"disp":False, "maxiter":200}):
    bounds = _distribution_bounds(dist_name)
    par_length = len(bounds['lower'])
    pvector_o  = _default_parameter_values(dist_name, x)
    pindex = list(range(0, par_length))
    if fixed is not None:
        pindex, pvector_o = _validate_fixed(dist_name, pindex, pvector_o, fixed)
    if len(pindex) == 0:
        raise ValueError(f'All parameters fixed. Estimation will not be carried out.')
    bounds['lower'][0] = -100 * abs(pvector_o[0])
    bounds['upper'][0] =  100 * abs(pvector_o[0])
    bounds['upper'][1] =  10 * abs(pvector_o[1])
    lbounds = [bounds['lower'][i] for i in pindex]
    ubounds = [bounds['upper'][i] for i in pindex]
    lbounds = np.atleast_1d(lbounds)
    ubounds = np.atleast_1d(ubounds)
    pvector_o = torch.tensor(pvector_o)
    init = pvector_o[pindex].numpy()
    b = Bounds(lb = lbounds.tolist(), ub = ubounds.tolist())
    scaler = torch.ones(par_length)
    fun = globals()[f'fd_d{dist_name}']
    objective_fun = lambda pars: fun(torch.tensor(pars, requires_grad=False), x, pvector_o, pindex, scaler, out = 'lik')
    sol = minimize(fun = objective_fun, x0 = init, method = method, jac = False, bounds = b, tol = tol, options = options)
    H = get_hessian(objective_fun, sol.x, lbounds.tolist(), ubounds.tolist())
    scaler = np.ones(par_length)
    scaler[pindex] = np.sqrt(1.0/np.abs(np.diag(H)))
    lbounds_scaled = np.ones(np.size(lbounds))
    lbounds_scaled = lbounds * 1.0/scaler[np.array(pindex)]
    ubounds_scaled = np.ones(np.size(ubounds))
    ubounds_scaled = ubounds * 1.0/scaler[np.array(pindex)]
    init_scaled = np.array(sol.x) * 1.0/scaler[np.array(pindex)]
    b_scaled = Bounds(lb = lbounds_scaled.tolist(), ub = ubounds_scaled.tolist())
    scaler = torch.tensor(scaler)
    objective_fun = lambda pars: fun(torch.tensor(pars, requires_grad=False), x, pvector_o, pindex, scaler, out = 'lik')
    sol = minimize(fun = objective_fun, x0 = init_scaled, method = method, jac = False, bounds = b_scaled, tol = tol, options = options)
    hess = get_hessian(objective_fun, sol.x, lbounds_scaled.tolist(), ubounds_scaled.tolist())
    jac_fun = lambda pars: fun(torch.tensor(pars, requires_grad=False), x, pvector_o, pindex, scaler, out = 'jac')
    scores = get_jacobian(jac_fun, sol.x, lbounds_scaled.tolist(), ubounds_scaled.tolist())
    parameters = pvector_o.numpy()
    index = np.array(pindex)
    parameters[index] = sol.x * scaler.numpy()[index]
    hess = hess/scaler.numpy()[index]
    D = np.diag(1.0/scaler.numpy()[index])
    hess = D.transpose() @ hess @ D
    scores = scores/scaler.numpy()[index]
    result = {'parameters':parameters, 'scaler':scaler.numpy()[index], 'index':index, 'hessian':hess, 'scores':scores, 'sol':sol}
    return result

def _adestimation(dist_name: str, x: Vector, fixed: Optional[Dict[str,float]] = None, method:str = 'L-BFGS-B', tol:float = 1e-08, options:dict = {"disp":False, "maxiter":200}):
    bounds = _distribution_bounds(dist_name)
    par_length = len(bounds['lower'])
    pvector_o  = _default_parameter_values(dist_name, x)
    pindex = list(range(0, par_length))
    if fixed is not None:
        pindex, pvector_o = _validate_fixed(dist_name, pindex, pvector_o, fixed)
    if len(pindex) == 0:
        raise ValueError(f'All parameters fixed. Estimation will not be carried out.')
    bounds['lower'][0] = -100 * abs(pvector_o[0])
    bounds['upper'][0] =  100 * abs(pvector_o[0])
    bounds['upper'][1] =  10 * abs(pvector_o[1])
    lbounds = [bounds['lower'][i] for i in pindex]
    ubounds = [bounds['upper'][i] for i in pindex]
    lbounds = np.atleast_1d(lbounds)
    ubounds = np.atleast_1d(ubounds)
    pvector_o = torch.tensor(pvector_o)
    init = pvector_o[pindex].numpy()
    b = Bounds(lb = lbounds.tolist(), ub = ubounds.tolist())
    scaler = torch.ones(par_length)
    fun = globals()[f'ad_d{dist_name}']
    objective_fun = lambda pars: fun(torch.tensor(pars, requires_grad=True), x, pvector_o, pindex, scaler, out = 'lik')
    sol = minimize(fun = objective_fun, x0 = init, method = method, jac = True, bounds = b, tol = tol, options = options)
    hess_func = lambda pars: fun(pars, x = x, pvector_o = pvector_o, pindex = pindex, scaler = scaler, out = 'hess')
    solx = torch.tensor(sol.x)
    hess_fun = torch.func.hessian(hess_func)
    H = hess_fun(solx)
    scaler = np.ones(par_length)
    scaler[pindex] = np.sqrt(1.0/np.abs(np.diag(H.numpy())))
    lbounds_scaled = np.ones(np.size(lbounds))
    lbounds_scaled = lbounds * 1.0/scaler[np.array(pindex)]
    ubounds_scaled = np.ones(np.size(ubounds))
    ubounds_scaled = ubounds * 1.0/scaler[np.array(pindex)]
    init_scaled = sol.x * 1.0/scaler[np.array(pindex)]
    b_scaled = Bounds(lb = lbounds_scaled.tolist(), ub = ubounds_scaled.tolist())
    scaler = torch.tensor(scaler)
    objective_fun = lambda pars: fun(torch.tensor(pars, requires_grad=True), x, pvector_o, pindex, scaler, out = 'lik')
    sol = minimize(fun = objective_fun, x0 = init_scaled, method = method, jac = True, bounds = b_scaled, tol = tol, options = options)
    hess_func = lambda pars: fun(pars, x = x, pvector_o = pvector_o, pindex = pindex, scaler = scaler, out = 'hess')
    solx = torch.tensor(sol.x)
    hess_fun = torch.func.hessian(hess_func)
    hess = hess_fun(solx)
    jac_func = lambda pars: fun(pars, x = x, pvector_o = pvector_o, pindex = pindex, scaler = scaler, out = 'jac')
    solx = torch.tensor(sol.x)
    scores = torch.autograd.functional.jacobian(jac_func, solx)
    parameters = pvector_o.numpy()
    index = np.array(pindex)
    parameters[index] = sol.x * scaler.numpy()[index]
    D = np.diag(1.0/scaler.numpy()[index])
    hess = D.transpose() @ hess.numpy() @ D
    scores = scores.numpy()/scaler.numpy()[index]
    result = {'parameters':parameters, 'scaler':scaler.numpy()[index], 'index':index, 'hessian':hess, 'scores':scores, 'sol':sol}
    return result

def estimate_distribution(dist_name: str, x: Vector, fixed: Optional[Dict[str,float]] = None, method:str = 'L-BFGS-B', tol:float = 1e-08, options:dict = {"disp":False, "maxiter":200}, type:str = 'AD') -> Dict[str, Any]:
    if type == 'AD':
        result = _adestimation(dist_name, x, fixed, method, tol, options)
    else:
        result = _fdestimation(dist_name, x, fixed, method, tol, options)
    return result


def eval_distribution(dist_name: str, x: Vector, type:str = 'AD', out = 'lik') -> Callable:
    if type == 'AD':
        fun = globals()[f'ad_d{dist_name}']
    else:
        fun = globals()[f'fd_d{dist_name}']
    bounds = _distribution_bounds(dist_name)
    par_length = len(bounds['lower'])
    pvector_o  = _default_parameter_values(dist_name, x)
    pindex = list(range(0, par_length))
    pvector_o = torch.tensor(pvector_o, dtype = torch.float64)
    init = pvector_o[pindex].numpy()
    scaler = torch.ones(par_length, dtype = torch.float64)
    if type == 'AD':
        objective_fun = lambda pars: fun(torch.tensor(pars, dtype = torch.float64, requires_grad=True), x, pvector_o, pindex, scaler, out = out)
    else:
        objective_fun = lambda pars: fun(torch.tensor(pars, dtype = torch.float64, requires_grad=False), x, pvector_o, pindex, scaler, out = out)
    return objective_fun


def _scale_fun(x, mu, sigma):
    return (x - mu)/sigma

def ad_djsu(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector = pvector * scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    rtau = 1.0/pvector[3]
    if torch.lt(rtau, torch.tensor(1e-7)):
        w = torch.tensor(1.0)
    else:
        w = torch.exp(torch.pow(rtau, 2.0))
    omega = -1.0 * pvector[2] * rtau
    c = torch.sqrt(1.0/(0.5 * (w - 1.0) * (w * torch.cosh(2.0 * omega) + 1.0)))
    z = (x - (c * torch.sqrt(w) * torch.sinh(omega)))/c
    r = - 1.0 * pvector[2] + torch.arcsinh(z)/rtau
    pdf = -1.0 * torch.log(c) - torch.log(rtau) - 0.5 * torch.log(torch.pow(z,2.0) + 1.0) - 0.5 * torch.log(torch.tensor(torch.pi * 2.0)) - 0.5 * torch.pow(r, 2.0)
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]
    elif out == 'hess':
        pdf = -1.0 * pdf.sum()
        return pdf
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf
    else:
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]

def fd_djsu(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector = pvector * scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    rtau = 1.0/pvector[3]
    if torch.lt(rtau, torch.tensor(1e-7)):
        w = torch.tensor(1.0)
    else:
        w = torch.exp(torch.pow(rtau, 2.0))
    omega = -1.0 * pvector[2] * rtau
    c = torch.sqrt(1.0/(0.5 * (w - 1.0) * (w * torch.cosh(2.0 * omega) + 1.0)))
    z = (x - (c * torch.sqrt(w) * torch.sinh(omega)))/c
    r = - 1.0 * pvector[2] + torch.arcsinh(z)/rtau
    pdf = -1.0 * torch.log(c) - torch.log(rtau) - 0.5 * torch.log(torch.pow(z,2.0) + 1.0) - 0.5 * torch.log(torch.tensor(torch.pi * 2.0)) - 0.5 * torch.pow(r, 2.0)
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        return pdf.item()
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf.numpy()
    else:
        pdf = -1.0 * pdf.sum()
        return pdf.item()


def _paramghst(betabar, nu):
    delta = torch.sqrt(1.0/(((2.0 * torch.pow(betabar,2.0))/(torch.pow(torch.subtract(nu,2.0), 2.0) * torch.subtract(nu,4.0))) + (1.0/torch.subtract(nu,2.0))))
    beta = betabar / delta
    mu = -1.0 * ((beta * torch.pow(delta, 2.0)) / torch.subtract(nu,2.0))
    params =  torch.tensor([mu, delta, beta, nu])
    return params

def fd_dsghst(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector = pvector * scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    params = _paramghst(pvector[2],pvector[3])
    beta_squared = torch.pow(params[2], 2.0)
    delta_squared = torch.pow(params[1], 2.0)
    res_squared = torch.pow(x - params[0], 2.0)
    pdf = ((1.0 - params[3])/2.0) * torch.log(torch.tensor(2.0)) + params[3] * torch.log(params[1]) + ((params[3] + 1.0)/2.0)\
          * torch.log(torch.abs(params[2])) + torch.log(torch_kve((params[3] + 1.0)/2.0,  torch.sqrt(beta_squared * (delta_squared + res_squared))))\
          - torch.sqrt(beta_squared * (delta_squared + res_squared)) + params[2] * (x - params[0]) - torch.lgamma(params[3]/2.0) - torch.log(torch.tensor(torch.pi))/2.0\
              - ((params[3] + 1.0)/2.0) * torch.log(delta_squared + res_squared)/2.0
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        return pdf.item()
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf.numpy()
    else:
        pdf = -1.0 * pdf.sum()
        return pdf.item()

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

def fd_dsgh(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector = pvector * scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    params = _paramgh(pvector[2],pvector[3], pvector[4])
    arg1 = params[2] * torch.sqrt(torch.pow(params[0], 2.0) - torch.pow(params[1], 2.0))
    a = (params[4]/2.0) * torch.log(torch.pow(params[0],2.0) - torch.pow(params[1],2.0))\
          - (torch.log(torch.sqrt(torch.tensor(2.0 * torch.pi))) + (params[4] - 0.5) * torch.log(params[0])\
             + params[4] * torch.log(params[2]) + torch.log(torch_kve(params[4], arg1)) - arg1)
    f = ((params[4] - 0.5)/2.0) * torch.log(torch.pow(params[2],2.0) + torch.pow((x - params[3]),2.0))
    arg2 = params[0] * torch.sqrt(torch.pow(params[2],2.0) + torch.pow((x - params[3]), 2.0))
    k = torch.log(torch_kve(params[4] - 0.5, arg2)) - arg2
    e = params[1] * (x - params[3])
    pdf = a + f + k + e
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        return pdf.item()
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf.numpy()
    else:
        pdf = -1.0 * pdf.sum()
        return pdf.item()

def _fs_skew_moments(m, skew):
    m_squared = torch.pow(m, 2.0)
    skew_squared = torch.pow(skew, 2.0)
    mu = m * torch.subtract(skew, torch.divide(torch.tensor(1.0),skew))
    sigma = torch.sqrt((1. - m_squared) * (skew_squared + 1.0/skew_squared) + 2.0 * m_squared - 1.0)
    out = [mu, sigma]
    return out


def ad_dsnorm(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector = pvector * scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    m = torch.divide(torch.tensor(2.0),torch.sqrt(torch.tensor(2.0 * torch.pi)))
    fs_mu, fs_sigma = _fs_skew_moments(m, pvector[2])
    z = x * fs_sigma + fs_mu
    z_sign = torch.sign(z)
    xi = torch.pow(pvector[2], z_sign)
    g = 2.0/(pvector[2] + 1.0/pvector[2])
    pdf = torch.log(g) + dnorm(z/xi) + torch.log(fs_sigma)
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]
    elif out == 'hess':
        pdf = -1.0 * pdf.sum()
        return pdf
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf
    else:
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]

def fd_dsnorm(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    m = torch.divide(torch.tensor(2.0),torch.sqrt(torch.tensor(2.0 * torch.pi)))
    fs_mu, fs_sigma = _fs_skew_moments(m, pvector[2])
    z = x * fs_sigma + fs_mu
    z_sign = torch.sign(z)
    xi = torch.pow(pvector[2], z_sign)
    g = 2.0/(pvector[2] + 1.0/pvector[2])
    pdf = torch.log(g)  + dnorm(z/xi) + torch.log(fs_sigma)
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        return pdf.item()
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf.numpy()
    else:
        pdf = -1.0 * pdf.sum()
        return pdf.item()

def fd_dged(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    lam = 0.5 * ((2./pvector[2]) * torch.log(torch.tensor(0.5)) + torch.lgamma(1.0/pvector[2]) - torch.lgamma(3.0/pvector[2]))
    g = torch.log(pvector[2]) - (lam + (1. + (1./pvector[2])) * torch.log(torch.tensor(2.)) + torch.lgamma(1/pvector[2]))
    pdf = g + (-0.5 * torch.pow(torch.abs(x/lam.exp()), pvector[2]))
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        return pdf.item()
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf.numpy()
    else:
        pdf = -1.0 * pdf.sum()
        return pdf.item()

def ad_dged(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    lam = 0.5 * ((2./pvector[2]) * torch.log(torch.tensor(0.5)) + torch.lgamma(1.0/pvector[2]) - torch.lgamma(3.0/pvector[2]))
    g = torch.log(pvector[2]) - (lam + (1. + (1./pvector[2])) * torch.log(torch.tensor(2.)) + torch.lgamma(1/pvector[2]))
    pdf = g + (-0.5 * torch.pow(torch.abs(x/lam.exp()), pvector[2]))
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]
    elif out == 'hess':
        pdf = -1.0 * pdf.sum()
        return pdf
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf
    else:
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]

def ad_dstd(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    scale = 0.5 * (torch.log(pvector[2]) - torch.log(pvector[2] - 2.0))
    x = x * scale.exp()
    alpha = torch.lgamma((pvector[2] + 1.0)/2.0) - 0.5 * (torch.log(torch.tensor(torch.pi)) + torch.log(pvector[2]))
    beta = torch.lgamma(pvector[2]/2.0) + 0.5 * (pvector[2] + 1.0) * torch.log((1.0 + torch.square(x)/pvector[2]))
    ratio = alpha - beta
    pdf = scale + ratio - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]
    elif out == 'hess':
        pdf = -1.0 * pdf.sum()
        return pdf
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf
    else:
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]

def fd_dstd(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    scale = 0.5 * (torch.log(pvector[2]) - torch.log(pvector[2] - 2.0))
    x = x * scale.exp()
    alpha = torch.lgamma((pvector[2] + 1.0)/2.0) - 0.5 * (torch.log(torch.tensor(torch.pi)) + torch.log(pvector[2]))
    beta = torch.lgamma(pvector[2]/2.0) + 0.5 * (pvector[2] + 1.0) * torch.log((1.0 + torch.square(x)/pvector[2]))
    ratio = alpha - beta
    pdf = scale + ratio - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        return pdf.item()
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf.numpy()
    else:
        pdf = -1.0 * pdf.sum()
        return pdf.item()

def fd_dnorm(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    pdf = torch.distributions.normal.Normal(pvector[0], pvector[1]).log_prob(x)
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        return pdf.item()
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf.numpy()
    else:
        pdf = -1.0 * pdf.sum()
        return pdf.item()

def ad_dnorm(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    pdf = torch.distributions.normal.Normal(pvector[0],pvector[1]).log_prob(x)
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]
    elif out == 'hess':
        pdf = -1.0 * pdf.sum()
        return pdf
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf
    else:
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]

def fd_dsged(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    lam = torch.sqrt(torch.pow(2.0,-2.0/pvector[3]) * torch.lgamma(1.0/pvector[3]).exp()/torch.lgamma(3.0/pvector[3]).exp())
    m = torch.pow(2.0, (1.0/pvector[3])) * lam * torch.lgamma(2.0/pvector[3]).exp()/torch.lgamma(1.0/pvector[3]).exp()
    fs_mu, fs_sigma = _fs_skew_moments(m, pvector[2])
    z = x * fs_sigma + fs_mu
    z_sign = torch.sign(z)
    xi = torch.pow(pvector[2], z_sign)
    g = 2.0/(pvector[2] + 1.0/pvector[2])
    new_pars = torch.cat([torch.tensor([0., 1.]), pvector[3].unsqueeze(0)])
    ged_log_lik = ad_dged(parameters = new_pars, x = z/xi, pvector_o = torch.ones(3, dtype = torch.float64), pindex = [0,1,2], scaler = torch.ones(3, dtype = torch.float64), out = 'jac')
    pdf = torch.log(g) + (torch.tensor(-1.0) * ged_log_lik) + torch.log(fs_sigma)
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        return pdf.item()
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf.numpy()
    else:
        pdf = -1.0 * pdf.sum()
        return pdf.item()

def ad_dsged(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    lam = torch.sqrt(torch.pow(2.0,-2.0/pvector[3]) * torch.lgamma(1.0/pvector[3]).exp()/torch.lgamma(3.0/pvector[3]).exp())
    m = torch.pow(2.0, (1.0/pvector[3])) * lam * torch.lgamma(2.0/pvector[3]).exp()/torch.lgamma(1.0/pvector[3]).exp()
    fs_mu, fs_sigma = _fs_skew_moments(m, pvector[2])
    z = x * fs_sigma + fs_mu
    z_sign = torch.sign(z)
    xi = torch.pow(pvector[2], z_sign)
    g = 2.0/(pvector[2] + 1.0/pvector[2])
    new_pars = torch.cat([torch.tensor([0., 1.]), pvector[3].unsqueeze(0)])
    ged_log_lik = ad_dged(parameters = new_pars, x = z/xi, pvector_o = torch.ones(3, dtype = torch.float64), pindex = [0,1,2], scaler = torch.ones(3, dtype = torch.float64), out = 'jac')
    pdf = torch.log(g) + (torch.tensor(-1.0) * ged_log_lik) + torch.log(fs_sigma)
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]
    elif out == 'hess':
        pdf = -1.0 * pdf.sum()
        return pdf
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf
    else:
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]


def fd_dsstd(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    m = 2.0 * torch.sqrt(pvector[3] - 2.0)/(pvector[3] - 1.0)/torch.exp(betaln(torch.tensor(0.5), pvector[3]/2.0))
    fs_mu, fs_sigma = _fs_skew_moments(m, pvector[2])
    z = x * fs_sigma + fs_mu
    z_sign = torch.sign(z)
    xi = torch.pow(pvector[2], z_sign)
    g = 2.0/(pvector[2] + 1.0/pvector[2])
    new_pars = torch.cat([torch.tensor([0., 1.]), pvector[3].unsqueeze(0)])
    std_log_lik = ad_dstd(parameters = new_pars, x = z/xi, pvector_o = torch.ones(3, dtype = torch.float64), pindex = [0,1,2], scaler = torch.ones(3, dtype = torch.float64), out = 'jac')
    pdf = torch.log(g) + (torch.tensor(-1.0) * std_log_lik) + torch.log(fs_sigma)
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        return pdf.item()
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf.numpy()
    else:
        pdf = -1.0 * pdf.sum()
        return pdf.item()


def ad_dsstd(parameters, x, pvector_o, pindex, scaler, out:"str" = 'lik'):
    pvector = pvector_o.clone()
    pvector[pindex] = parameters
    pvector *= scaler
    x = _scale_fun(x, pvector[0], pvector[1])
    m = 2.0 * torch.sqrt(pvector[3] - 2.0)/(pvector[3] - 1.0)/torch.exp(betaln(torch.tensor(0.5), pvector[3]/2.0))
    fs_mu, fs_sigma = _fs_skew_moments(m, pvector[2])
    z = x * fs_sigma + fs_mu
    z_sign = torch.sign(z)
    xi = torch.pow(pvector[2], z_sign)
    g = 2.0/(pvector[2] + 1.0/pvector[2])
    new_pars = torch.cat([torch.tensor([0., 1.]), pvector[3].unsqueeze(0)])
    std_log_lik = ad_dstd(parameters = new_pars, x = z/xi, pvector_o = torch.ones(3, dtype = torch.float64), pindex = [0,1,2], scaler = torch.ones(3, dtype = torch.float64), out = 'jac')
    pdf = torch.log(g) + (torch.tensor(-1.0) * std_log_lik) + torch.log(fs_sigma)
    pdf = pdf - torch.log(pvector[1])
    if out == 'lik':
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]
    elif out == 'hess':
        pdf = -1.0 * pdf.sum()
        return pdf
    elif out == 'jac':
        pdf = -1.0 * pdf
        return pdf
    else:
        pdf = -1.0 * pdf.sum()
        pdf.backward()
        grad_np = np.asanyarray(parameters.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]
