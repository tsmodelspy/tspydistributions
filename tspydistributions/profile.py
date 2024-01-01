from __future__ import annotations
from tspydistributions.estimation import estimate_distribution
from typing import List, Dict, Literal, Any, Optional, TYPE_CHECKING, TypeVar, Callable, Tuple
import tspydistributions.pdqr as pdqr
from tspydistributions.helpers import _validate_distribution, _distribution_bounds, _distribution_parvector
import numpy as np
import numpy.typing as npt
import torch
from multiprocessing import Pool
from os import cpu_count

Array = npt.NDArray[np.float64]
Vector = npt.ArrayLike

def worker_function(args):
    dist_name, mu, sigma, skew, shape, lamda, n = args
    x = _random_generator(n, dist_name, mu, sigma, skew, shape, lamda)
    mod = estimate_distribution(dist_name, x)
    return mod['parameters'].tolist()

def _random_generator(n = 100, dist_name:str = 'norm', mu:float = 0, sigma:float = 1, skew:float = 0, shape:float = 5, lamda:float = -0.5):
    x = 0.0
    match dist_name:
        case "norm":
            x = pdqr.rnorm(n, mu, sigma)    
        case "std":
            x = pdqr.rstd(n, mu, sigma, shape)
        case "ged":
            x = pdqr.rged(n, mu, sigma, shape)
        case "snorm":
            x = pdqr.rsnorm(n, mu, sigma, skew)
        case "sged":
            x = pdqr.rsged(n, mu, sigma, skew, shape)
        case "sstd":
            x = pdqr.rsstd(n, mu, sigma, skew, shape)
        case "jsu":
            x = pdqr.rjsu(n, mu, sigma, skew, shape)
        case "sgh":
            x = pdqr.rsgh(n, mu, sigma, shape, lamda)
        case "sghst":
            x = pdqr.rsghst(n, mu, sigma, skew, shape)
        case _:
            x = pdqr.rstd(n, mu, sigma, shape)
    x = torch.tensor(x, dtype = torch.float64)
    return x

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def profile_distribution(dist_name:str = 'norm', mu:float = 0, sigma:float = 1, skew:float = 0, shape:float = 5, lamda:float = -0.5, sim:int  = 100, size = [100, 200, 400, 800, 1000, 1500, 2000, 4000], num_workers:Optional[int] = None)->Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    if _validate_distribution(dist_name) == False:
        raise ValueError("not a valid distribution")
    bounds = _distribution_bounds(dist_name)
    par_length = len(bounds['lower'])
    results = np.empty((len(size), sim, par_length))
    pars = _distribution_parvector(dist_name, mu, sigma, skew, shape, lamda)
    if num_workers is None:
        num_workers = cpu_count()  # Defaults to the number of available CPU cores

    with Pool(processes=num_workers) as pool:
        for size_idx, n in enumerate(size):
            # Prepare arguments for each simulation
            args_list = [(dist_name, mu, sigma, skew, shape, lamda, n) for _ in range(sim)]            
            # Perform parallel computation
            parallel_results = pool.map(worker_function, args_list)
            # Store the results
            for sim_idx, pars in enumerate(parallel_results):
                results[size_idx, sim_idx, :] = pars

    results_dict = {size_val: results[idx, :, :] for idx, size_val in enumerate(size)}
    rmse_results = {}
    
    for size_val, simulations in results_dict.items():
        # Initialize an array to store the RMSE for each parameter for this size
        rmse_per_param = np.zeros(par_length)
        for param_index in range(par_length):
            # Calculate RMSE for each parameter
            rmse_per_param[param_index] = rmse(simulations[:, param_index], pars[param_index])
        # Store RMSE results for this size
        rmse_results[size_val] = rmse_per_param
    
    return results_dict, rmse_results
