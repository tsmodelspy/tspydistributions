from typing import Dict, List, Optional
import warnings
from tspydistributions.sandwich import vcov
import numpy as np
import numpy.typing as npt
Vector = npt.ArrayLike
Array = npt.NDArray[np.float64]

def _sublist(lst: List, indices: List[int]) -> List:
    return [lst[i] for i in indices]


def _get_parameters(name: str, mu: float, sigma: float, skew: Optional[float], shape: Optional[float], lamda: Optional[float]) -> List[float]:
    par_vector = [mu, sigma]
    match name:
        case "norm":
            par_vector = [mu, sigma]
        case "std":
            par_vector =  [mu, sigma, shape]
        case "ged":
            par_vector =  [mu, sigma, shape]
        case "snorm":
            par_vector =  [mu, sigma, skew]
        case "sstd":
            par_vector =  [mu, sigma, skew, shape]
        case "sged":
            par_vector =  [mu, sigma, skew, shape]
        case "jsu":
            par_vector =  [mu, sigma, skew, shape]
        case "ghst":
            par_vector =  [mu, sigma, skew, shape]
        case "sgh":
            par_vector =  [mu, sigma, skew, shape, lamda]
    return par_vector
 
def _distribution_bounds(name: str = "norm")-> Dict[str,list[float]]:
    bounds = {"lower": [-float('inf'), float(1e-12)], "upper" : [float('inf'), float('inf')]}
    match name:
        case "norm":
            bounds = {"lower" : [-float('inf'), float(1e-12)], "upper" : [float('inf'), float('inf')]}
        case "std":
            bounds = {"lower" : [-float('inf'), float(1e-12), float(4.001)], "upper" : [float('inf'), float('inf'), float(100)]}
        case "ged":
            bounds = {"lower" : [-float('inf'), float(1e-12), float(0.1)], "upper" : [float('inf'), float('inf'), float(100)]}
        case "snorm":
            bounds = {"lower" : [-float('inf'), float(1e-12), float(0.01)], "upper" : [float('inf'), float('inf'), float(20)]}
        case "sged":
            bounds = {"lower" : [-float('inf'), float(1e-12), float(0.01), float(0.1)], "upper" : [float('inf'), float('inf'), float(30), float(100)]}
        case "sstd":
            bounds = {"lower" : [-float('inf'), float(1e-12), float(0.01), float(2.01)], "upper" : [float('inf'), float('inf'), float(30), float(100)]}
        case "jsu":
            bounds = {"lower" : [-float('inf'), float(1e-12), float(-20.0), float(0.1)], "upper" : [float('inf'), float('inf'), float(20), float(100)]}
        case "sghst":
            bounds = {"lower" : [-float('inf'), float(1e-12), float(-80.0), float(4.01)], "upper" : [float('inf'), float('inf'), float(80), float(100)]}
        case "sgh":
            bounds = {"lower" : [-float('inf'), float(1e-12), float(-0.999), float(0.25), float(-6.0)], "upper" : [float('inf'), float('inf'), float(0.99), float(100), float(6.0)]}

    return bounds

            
def _default_parameter_values(dist_name: str, x:Optional[Vector] = None)->List[float]:
    if x is not None:
        x = np.asarray(x)
        mu = np.sum(x)/len(x)
        sigma = sum((x - mu)**2 for x in x)/len(x)
        sigma = sigma ** 0.5
    else:
        mu = 0.0
        sigma = 1.0
    defaults = [mu, sigma]
    if dist_name == 'std':
        defaults = [mu, sigma, 4.2]
    if dist_name == 'ged':
        defaults = [mu, sigma, 0.5]
    if dist_name == 'snorm':
        defaults = [mu, sigma, 2]
    if dist_name == 'sged':
        defaults = [mu, sigma, 2, 0.5]
    if dist_name == 'sstd':
        defaults = [mu, sigma, 2, 5]
    if dist_name == 'jsu':
        defaults = [mu, sigma, 2, 0.5]
    if dist_name == 'sghst':
        defaults = [mu, sigma, 1, 4.2]
    if dist_name == 'sgh':
        defaults = [mu, sigma, 0.5, 0.5, -0.5]
    return defaults


def _distribution_par_names(dist_name) -> List[str]:
    dnames = ['mu','sigma']
    if dist_name == 'norm':
        dnames =  ['mu','sigma']
    if dist_name in ['std','ged']:
        dnames = ['mu','sigma','shape']
    if dist_name == 'snorm':
        dnames =  ['mu','sigma','skew']
    if dist_name in ['sged','sstd','jsu','sghst']:
        dnames =  ['mu','sigma','skew','shape']
    if dist_name == 'sgh':
        dnames =  ['mu','sigma','skew','shape','lamda']
    return dnames

def _distribution_parvector(dist_name:str = 'norm', mu:float = 0, sigma:float = 1, skew:float = 0, shape:float = 5, lamda:float = -0.5) -> Array:
    dvalue = [mu, sigma]
    if dist_name == 'norm':
        dvalue =  [mu,sigma]
    if dist_name in ['std','ged']:
        dvalue = [mu,sigma,shape]
    if dist_name == 'snorm':
        dvalue =  [mu,sigma,skew]
    if dist_name in ['sged','sstd','jsu','sghst']:
        dvalue =  [mu,sigma,skew,shape]
    if dist_name == 'sgh':
        dvalue =  [mu,sigma,skew,shape,lamda]
    dvalue = np.asarray(dvalue)
    return dvalue

def valid_distributions() -> List[str]:
    dist = ['norm','std','ged','snorm','sged','sstd','sgh','sghst','jsu']
    return dist

def _validate_distribution(dist_name = 'norm')->bool:
    valid = valid_distributions()
    if dist_name not in valid:
        return False
    else:
        return True

def _distname_jdist(dist_name = 'norm') -> str:
    if _validate_distribution(dist_name) == False:
        raise ValueError(f'{dist_name} is not a valid distribution')
    else:
        dname = f'jd{dist_name}'
        return dname

def _validate_fixed(dist_name:str, pindex:List[int], pvector:List[float], fixed:Dict[str,float])->List:
    fixed_names = list(fixed.keys())
    valid_parameters = _distribution_par_names(dist_name)
    n = len(fixed_names)
    for i in range(n):
        if fixed_names[i] not in valid_parameters:
            warnings.warn(f'{fixed_names[i]} is not a valid parameter for the {dist_name} distribution')
        else:
            if fixed_names[i] == 'mu':
                pindex[0] = int(999)
                pvector[0] = fixed['mu']
            if fixed_names[i] == 'sigma':
                pindex[1] = int(999)
                pvector[1] = fixed['sigma']
            if fixed_names[i] == 'skew':
                pindex[2] = int(999)
                pvector[2] = fixed['skew']
            if fixed_names[i] == 'shape' and dist_name in ['std','ged']:
                pindex[2] = int(999)
                pvector[2] = fixed['shape']
            if fixed_names[i] == 'shape' and dist_name not in ['std','ged']:
                pindex[3] = int(999)
                pvector[3] = fixed['shape']
            if fixed_names[i] == 'lamda':
                pindex[4] = int(999)
                pvector[4] = fixed['lamda']
    
    pindex = [i for i in pindex if i<999]
    return [pindex, pvector]



def _table(name, pvector, pindex, hess, scores, type = 'H'):
    from tspydistributions.pdqr import pnorm
    pnames = _distribution_par_names(name)
    v = vcov(hess, scores, type)
    std_errors = np.asarray([0] * len(pnames))
    std_errors[pindex] = np.sqrt(np.diag(v))
    t_values = np.asarray([np.nan] * len(pnames))
    t_values[pindex] = pvector[pindex]/std_errors[pindex]
    p_values = np.asarray([np.nan] * len(pnames))
    p_values[pindex] = 2.0 * (1.0 - pnorm(np.abs(t_values)))