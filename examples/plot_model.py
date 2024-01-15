"""
Demo: Linear Model with Johnson's SU-distribution
-------------------------------------------------
"""

import torch
import numpy as np
from scipy.optimize import minimize, Bounds
import tspydistributions.pdqr as pdqr
from tspydistributions.density import djsu
from tspydistributions.numderiv import get_hessian
from tspydistributions.helpers import _distribution_bounds
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

# generate some simulated data
alpha = 0.1
beta = 0.5
x = pdqr.rnorm(5000, mu = 0.01, sigma = 0.1, seed = 20)
y = alpha + beta * x + pdqr.rjsu(5000, mu = 0, sigma = 0.25, skew = 1, shape = 5, seed = 20)

# 1-stage estimation with no derivatives using the pdqr numpy functions
def nploglik(params, y, x):
    alpha, beta, sigma, skew, shape = params
    res = y - alpha - beta * x
    f = pdqr.djsu(res, mu = 0.0, sigma = sigma, skew = skew, shape = shape, log=True)
    return -np.sum(f)

bounds = _distribution_bounds('jsu')

lower = [float(-np.inf), float(-np.inf)] + bounds['lower'][1:4]
upper = [float(np.inf), float(np.inf)] + bounds['upper'][1:4]

result_1 = minimize(nploglik, x0 = np.array([0, 1, 1, 2, 2]), args = (y, x), method = 'SLSQP', bounds = Bounds(lower, upper), options={'maxiter': 1000, 'ftol':1e-12,'finite_diff_rel_step':'2-point'})
# calculate standard errors
hess_fun_1 = lambda pars: nploglik(pars, y, x)
hessian_1 = get_hessian(hess_fun_1, result_1.x, [-np.inf, -np.inf, 1e-12, -np.inf, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf])
std_errors_1 = np.sqrt(np.diag(np.linalg.inv(hessian_1)))


# 1-stage estimation with derivatives using the torch density functions

def adloglik(params, y, x, out = 'loglik'):
    #alpha, beta, sigma, skew, shape = params
    res = y - params[0] - params[1] * x
    parameters = torch.cat([torch.tensor([0.0]),params[2].unsqueeze(0),params[3].unsqueeze(0),params[4].unsqueeze(0)])
    f = djsu(parameters, res, log = True)
    if out == 'loglik':
        pdf = -1.0 * f.sum()
        pdf.backward()
        grad_np = np.asanyarray(params.grad.numpy(), dtype=np.float64)
        return [pdf.item(), grad_np]
    else:
        # for the hessian calculation
        return -1.0 * f.sum()

fun_2 = lambda pars: adloglik( params = torch.tensor(pars, dtype = torch.float64, requires_grad = True), y = torch.tensor(y), x = torch.tensor(x), out = 'loglik')
result_2 = minimize(fun_2, x0 = np.array([0, 1, 1, 2, 2]), jac=True, method = 'SLSQP', bounds = Bounds([-np.inf, -np.inf, 1e-12, -np.inf, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf]), options={'maxiter': 1000,'ftol':1e-12})
fun_2 = lambda pars: adloglik(params = pars, y = torch.tensor(y), x = torch.tensor(x), out = 'hessian')
hess_fun_2 = torch.func.hessian(fun_2)
optimal_pars = torch.tensor(result_2.x, dtype = torch.float64)
hessian_2 = hess_fun_2(optimal_pars)
std_errors_2 = np.sqrt(np.diag(np.linalg.inv(hessian_2)))

# collect results and print results
parameters_results = [
    ["True Parameters", 0.1, 0.5, 0.25, 1, 5],
    ["1-stage estimation with no derivatives", result_1.x[0], result_1.x[1], result_1.x[2], result_1.x[3], result_1.x[4]],
    ["1-stage estimation with derivatives", result_2.x[0], result_2.x[1], result_2.x[2], result_2.x[3], result_2.x[4]]
]

parameters_table_headers = ["Method", "Alpha", "Beta", "Sigma", "Skew", "Shape"]
parameters_table = tabulate(parameters_results, headers=parameters_table_headers, tablefmt="grid")

# Standard errors table
std_errors_results = [
    ["1-stage estimation with no derivatives", std_errors_1[0], std_errors_1[1], std_errors_1[2], std_errors_1[3], std_errors_1[4]],
    ["1-stage estimation with derivatives", std_errors_2[0], std_errors_2[1], std_errors_2[2], std_errors_2[3], std_errors_2[4]]
]

std_errors_table_headers = ["Method", "Std Error Alpha", "Std Error Beta", "Std Error Sigma", "Std Error Skew", "Std Error Shape"]
std_errors_table = tabulate(std_errors_results, headers=std_errors_table_headers, tablefmt="grid")

# Print tables
print("Parameters:")
print(parameters_table)
print("\nStandard Errors:")
print(std_errors_table)