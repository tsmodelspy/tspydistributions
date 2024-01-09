from __future__ import annotations
from dataclasses import dataclass, field
from tspydistributions.helpers import _distribution_bounds, _distribution_par_names
import tspydistributions.pdqr as pdqr
from tspydistributions.estimation import estimate_distribution
from tspydistributions.sandwich import vcov
from tspydistributions.profile import profile_distribution
from tspydistributions.plot import pdfplot, qqplot, qqline
from typing import ClassVar, List, Dict, Literal, Any, Optional, TYPE_CHECKING, TypeVar, Callable
from tabulate import tabulate
import numpy.typing as npt
import torch
import pandas as pd
import numpy as np
from scipy.integrate import quad
import inspect
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    labs
)

Array = npt.NDArray[np.float64]
Vector = npt.ArrayLike

def _format_list(x:list, decimals:int = 2):
    #format_string = "{:." + str(decimals) + "f}"
    #formatted_numbers = [format_string.format(i) for i in x]
    #return formatted_numbers
    return ["{:.{}f}".format(i, decimals) for i in x]

def print_table(left, right, width):
    key_value_sep = ": "
    column_gap = " " * 3
    gap_width = len(column_gap)
    width -= gap_width
    half_width = width // 2
    left_space = half_width - len(key_value_sep)
    right_space = half_width - len(key_value_sep) + int(width % 2 == 0)  # Add extra space if width is even
    divider = '=' * (width + gap_width)
    print(divider)
    max_rows = max(len(left), len(right))
    left_items = list(left.items())
    right_items = list(right.items())
    for i in range(max_rows):
        left_key, left_val = left_items[i] if i < len(left_items) else ("", "")
        right_key, right_val = right_items[i] if i < len(right_items) else ("", "")
        left_key = (left_key[:left_space - len(str(left_val))] + '...') if len(left_key) + len(str(left_val)) > left_space else left_key
        right_key = (right_key[:right_space - len(str(right_val))] + '...') if len(right_key) + len(str(right_val)) > right_space else right_key
        left_side = f"{left_key}{key_value_sep}{left_val:>{left_space - len(left_key)}}" if left_key else ""
        right_side = f"{right_key}{key_value_sep}{right_val:>{right_space - len(right_key)}}" if right_key else ""
        print(f"{left_side:<{half_width}}{column_gap}{right_side:<{half_width + (width % 2 == 0)}}")  # Print with extra space for even width


@dataclass(kw_only=True)
class Distribution():
    """Distribution Class

    All distributions are parameterized in terms of mean and standard deviation (sigma).
    
    :param name: the distribution name. Valid distribution are the Normal ('norm'), Student ('std') and Generalized Error ('ged') distributions; 
                 the skewed variants of these based on the transformations in :cite:p:`Fernandez1998`, which are the Skew Normal ('snorm'), 
                 Skew Student ('sstd') and Skew Generalized Error ('sged`) distributions; The reparameterized version of :cite:p:`Johnson1949`
                 SU distribution ('jsu'); the Generalized Hyperbolic ('sgh') distribution of :cite:p:`Barndorff1977` and the Generalized 
                 Hyperbolic Skew Student (`sghst`) distribution of :cite:p:`Aas2006`
    :param mu: the mean
    :param sigma: the standard deviation
    :param skew: the skew parameter
    :param shape: the shape parameter
    :param lamda: additional shape parameter for the Generalized Hyperbolic distribution
    """

    # get defaults for parameters, hessian, scores, loglik, nobs, success, data when not estimated
    name: str = "norm"
    mu: float = 0
    sigma: float = 1
    skew: float = 0.9
    shape: float = 5
    lamda: float = 0
    __bounds: Dict[str, List[float]] = field(init=False, repr=False, default_factory=dict)
    __original_bounds: Dict[str, List[float]] = field(init=False, repr=False, default_factory=dict)
    
    def __post_init__(self):
        self.__bounds = _distribution_bounds(self.name)
        self.__original_bounds = self.__bounds.copy()        
        # Ensure sigma > 0 when the object is initialized
        if self.sigma <= 0:
            raise ValueError("Sigma must be greater than 0.")

    def __setattr__(self, name, value):
        if name == "name":
            # If the name attribute is being set, we also update bounds
            object.__setattr__(self, "__bounds", _distribution_bounds(value))
            object.__setattr__(self, "__original_bounds", _distribution_bounds(value))
        elif name == "sigma":
            # If the sigma attribute is being set, we enforce that it's greater than 0
            if value <= 0:
                raise ValueError("sigma must be greater than 0")
        object.__setattr__(self, name, value)

    @property
    def bounds(self):
        return self.__bounds

    @bounds.setter
    def bounds(self, value):
        if all(k in value for k in ["lower", "upper"]):
            for i, (new_lower, orig_lower) in enumerate(zip(value["lower"], self.__original_bounds["lower"])):
                if new_lower < orig_lower:
                    raise ValueError(f"The new lower bound at index {i} violates the original bound.")
            
            for i, (new_upper, orig_upper) in enumerate(zip(value["upper"], self.__original_bounds["upper"])):
                if new_upper > orig_upper:
                    raise ValueError(f"The new upper bound at index {i} violates the original bound.")
        
        self.__bounds = value

    def __repr__(self):
        return (
            f"Distribution(\n"
            f"  name={self.name!r},\n"
            f"  mu={self.mu!r},\n"
            f"  sigma={self.sigma!r},\n"
            f"  skew={self.skew!r},\n"
            f"  shape={self.shape!r},\n"
            f"  lamda={self.lamda!r}\n"
            f")"
        )
    
    def cdf(self, q: Vector, mu: Optional[Vector] = None, sigma: Optional[Vector] = None, skew: Optional[Vector] = None, shape: Optional[Vector] = None, lamda: Optional[Vector] = None, lower_tail: bool = True) -> Array:
        """
        Cumulative Probability Function

        The distribution parameters are read from the class object if they are not None.

        :param q: a vector of quantiles
        :param mu: the mean
        :param sigma: the standard deviation
        :param skew: the skew parameter
        :param shape: the shape parameter
        :param lamda: the GH lamda parameter
        :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
        :rtype: a numpy array
        """
        fun = getattr(pdqr, f'p{self.name}')
        params = inspect.signature(fun).parameters
        args = {'q': q, 'lower_tail': lower_tail}
        if 'mu' in params:
            if mu is not None:
                args['mu'] = mu
            else:
                args['mu'] = self.mu
        if 'sigma' in params:
            if sigma is not None:
                args['sigma'] = sigma
            else:
                args['sigma'] = self.sigma
        if 'skew' in params:
            if skew is not None:
                args['skew'] = skew
            else:
                args['skew'] = self.skew
        if 'shape' in params:
            if shape is not None:
                args['shape'] = shape
            else:
                args['shape'] = self.shape
        if 'lamda' in params:
            if lamda is not None:
                args['lamda'] = lamda
            else:
                args['lamda'] = self.lamda
        return fun(**args)

    def pdf(self, x: Vector, mu: Optional[Vector] = None, sigma: Optional[Vector] = None, skew: Optional[Vector] = None, shape: Optional[Vector] = None, lamda: Optional[Vector] = None, log: bool = False) -> Array:
        """
        Probability Density Function

        The distribution parameters are read from the class object if they are not None.

        :param x: a vector of quantiles
        :param mu: the mean
        :param sigma: the standard deviation
        :param skew: the skew parameter
        :param shape: the shape parameter
        :param lamda: the GH lamda parameter
        :param log: whether to return the log density
        :rtype: a numpy array
        """
        fun = getattr(pdqr, f'd{self.name}')
        params = inspect.signature(fun).parameters
        args = {'x': x, 'log': log}
        if 'mu' in params:
            if mu is not None:
                args['mu'] = mu
            else:
                args['mu'] = self.mu
        if 'sigma' in params:
            if sigma is not None:
                args['sigma'] = sigma
            else:
                args['sigma'] = self.sigma
        if 'skew' in params:
            if skew is not None:
                args['skew'] = skew
            else:
                args['skew'] = self.skew
        if 'shape' in params:
            if shape is not None:
                args['shape'] = shape
            else:
                args['shape'] = self.shape
        if 'lamda' in params:
            if lamda is not None:
                args['lamda'] = lamda
            else:
                args['lamda'] = self.lamda
        return fun(**args)
    
    def quantile(self, p: Vector, mu: Optional[Vector] = None, sigma: Optional[Vector] = None, skew: Optional[Vector] = None, shape: Optional[Vector] = None, lamda: Optional[Vector] = None, lower_tail: bool = True) -> Array:
        """
        Quantile Function
        
        This method is also accessible via the `ppf` alias.

        The distribution parameters are read from the class object if they are not None.

        :param p: a vector of probabilities
        :param mu: the mean
        :param sigma: the standard deviation
        :param skew: the skew parameter
        :param shape: the shape parameter
        :param lamda: the GH lamda parameter
        :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
        :rtype: a numpy array
        """
        fun = getattr(pdqr, f'q{self.name}')
        params = inspect.signature(fun).parameters
        args = {'p': p, 'lower_tail': lower_tail}
        if 'mu' in params:
            if mu is not None:
                args['mu'] = mu
            else:
                args['mu'] = self.mu
        if 'sigma' in params:
            if sigma is not None:
                args['sigma'] = sigma
            else:
                args['sigma'] = self.sigma
        if 'skew' in params:
            if skew is not None:
                args['skew'] = skew
            else:
                args['skew'] = self.skew
        if 'shape' in params:
            if shape is not None:
                args['shape'] = shape
            else:
                args['shape'] = self.shape
        if 'lamda' in params:
            if lamda is not None:
                args['lamda'] = lamda
            else:
                args['lamda'] = self.lamda
        return fun(**args)
    
    # alias to conform with scipy.stats

    def ppf(self, p: Vector, mu: Optional[Vector] = None, sigma: Optional[Vector] = None, skew: Optional[Vector] = None, shape: Optional[Vector] = None, lamda: Optional[Vector] = None, lower_tail: bool = True) -> Array:
        """
        Alias method for quantile function
        """
        return self.quantile(p = p, mu = mu, sigma = sigma, skew = skew, shape = shape, lamda = lamda, lower_tail = lower_tail)
    

    def random(self, n:int = 1, mu:Optional[Vector] = None, sigma: Optional[Vector] = None, skew: Optional[Vector] = None, shape: Optional[Vector] = None, lamda: Optional[Vector] = None, seed: Optional[int] = None) -> Array:
        """
        Random Number Function

        The distribution parameters are read from the class object if they are not None.
       
        :param n: the number of draws
        :param mu: the mean
        :param sigma: the standard deviation
        :param skew: the skew parameter
        :param shape: the shape parameter
        :param lamda: the GH lamda parameter
        :param seed: an optional value to initialize the random seed generator
        :rtype: a numpy array
        """
        fun = getattr(pdqr, f'r{self.name}')
        params = inspect.signature(fun).parameters
        args = {'n': n, 'seed': seed}
        if 'mu' in params:
            if mu is not None:
                args['mu'] = mu
            else:
                args['mu'] = self.mu
        if 'sigma' in params:
            if sigma is not None:
                args['sigma'] = sigma
            else:
                args['sigma'] = self.sigma
        if 'skew' in params:
            if skew is not None:
                args['skew'] = skew
            else:
                args['skew'] = self.skew
        if 'shape' in params:
            if shape is not None:
                args['shape'] = shape
            else:
                args['shape'] = self.shape
        if 'lamda' in params:
            if lamda is not None:
                args['lamda'] = lamda
            else:
                args['lamda'] = self.lamda
        return fun(**args) 
    
    # alias for random to conform with scipy.stats
    def rvs(self, n:int = 1, mu: Optional[Vector] = None, sigma: Optional[Vector] = None, skew: Optional[Vector] = None, shape: Optional[Vector] = None, lamda: Optional[Vector] = None, seed: Optional[int] = None) -> Array:
        """
        Alias method for random function
        """
        return self.random(n = n, mu = mu, sigma = sigma, skew = skew, shape = shape, lamda = lamda, seed = seed)

    def skewness(self):
        """
        Distribution Skewness
        """
        return pdqr.skewness(self.name, self.mu, self.sigma, self.skew, self.shape, self.lamda)
    
    def kurtosis(self):
        """
        Distribution Kurtosis
        """
        return pdqr.kurtosis(self.name, self.mu, self.sigma, self.skew, self.shape, self.lamda)
    
    def estimate(self, x: Vector, fixed: Optional[Dict[str,float]] = None, method:str = 'L-BFGS-B', tol:float = 1e-08, options:dict = {"disp":False, "maxiter":200}, type:str = 'AD') -> 'EstimatedDistribution':
        """
        Parameter Estimation

        Given a vector x and optionally any fixed values, estimates the parameters of the distribution using the scipy minimize function
        and pytorch based gradient of the likelihood. The method returns an object of class `EstimatedDistribution` with additional methods
        for summary, vcov, coef etc. 
        For the sghst and sgh distribution, only type = 'FD' is supported until such time as the modified Bessel function of the second kind 
        is implemented in pytorch.

        :param x: a vector representing a stationary series
        :param fixed: an optional dictionary of name-value pairs which are fixed instead of estimated
        :param method: the scipy algorithm to use for estimation
        :param tol: termination tolerance
        :param options: a dictionary of options to pass to the scipy minimize function
        :param type: the type of numerical differentiation to use. Valid choices are 'FD' for finite differences and 'AD' for automatic differentiation
        :rtype: an object of class EstimatedDistribution
        """

        if self.name in ('sghst', 'sgh'):
            type = 'FD'
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype = torch.float64)

        out = estimate_distribution(self.name, x = x, fixed = fixed, method = method, tol = tol, options = options, type = type)
        parameter_names = _distribution_par_names(self.name)
        parameters = out['parameters'].tolist()
        k = 0
        if 'mu' in parameter_names:
            self.mu = parameters[k]
            k+=1
        if 'sigma' in parameter_names:
            self.sigma = parameters[k]
            k+=1
        if 'skew' in parameter_names:
            self.skew = parameters[k]
            k+=1
        if 'shape' in parameter_names:
            self.shape = parameters[k]
            k+=1
        if 'lamda' in parameter_names:
            self.lamda = parameters[k]
        
        estimated_distribution = EstimatedDistribution(
            name=self.name, 
            mu=self.mu,
            sigma=self.sigma,
            skew=self.skew,
            shape=self.shape,
            lamda=self.lamda,
            parameters = np.atleast_1d(parameters),
            hessian=out['hessian'],
            scores=out['scores'],
            scaler=out['scaler'],
            index=out['index'].tolist(),
            loglikelihood=-1.0 * out['sol'].fun,
            x=x.numpy(),
            no_obs=len(x),
            solution = out['sol']
        )

        return estimated_distribution
    
    # alias for estimiate to conform with scipy.stats
    def fit(self, x: Vector, fixed: Optional[Dict[str,float]] = None, method:str = 'L-BFGS-B', tol:float = 1e-08, options:dict = {"disp":False, "maxiter":200}, type:str = 'AD') -> 'EstimatedDistribution':
        """
        Alias method for estimate
        """
        return self.estimate(x = x, fixed = fixed, method = method, tol = tol, options = options, type = type)
    

    def profile(self, sim:int = 100, size:list = [100, 200, 400, 800, 1000, 1500, 2000, 4000], num_workers:Optional[int] = None) -> 'ProfiledDistribution':
        """
        Profile Distribution

        Given a distribution, estimates the parameters for a range of sample sizes and simulations. The method returns an object of class
        ProfiledDistribution with the estimated parameters and the root mean squared error (RMSE) of the estimates. The RMSE is calculated 
        as the square root of the mean squared error (MSE) of the estimates, where the MSE is the sum of the squared difference between 
        the estimated and true parameters for each simulation and sample size combination, divided by the number of simulations.

        :param sim: the number of simulations
        :param size: the sample sizes
        :param num_workers: the number of workers to use for parallel processing
        :rtype: an object of class ProfileDistribution
        """
        results, rmse =  profile_distribution(self.name, mu = self.mu, sigma = self.sigma, skew = self.skew, shape = self.shape, lamda = self.lamda, sim = sim, size = size, num_workers = num_workers)
        profiled_distribution = ProfiledDistribution(
            name=self.name, 
            mu=self.mu,
            sigma=self.sigma,
            skew=self.skew,
            shape=self.shape,
            lamda=self.lamda,
            dist = results,
            rmse = rmse,
            sim = sim,
            size = size
        )
        return profiled_distribution
        

    def expectation(self, fun_str:str = "np.abs(x)", type:str = "d", lower:float = -np.inf, upper:float = np.inf):
        """
        Expectation of a custom function over either the pdf, cdf or quantile

        Given a custom numpy function on x, evaluates and returns the expectation based on the given bounds and distribution
        parameters, using numerical quadrature.

        :param fun_str: a valid string representing a function of x for which the expectation will be calculated given the distribution.
        :param type: valid choices are d, p and q representing the pdf, cdf and quantile functions.
        :param lower: the lower bound for the integral
        :param upper: the upper bound for the integral 
        :rtype: the expectation
        """
        if type not in ("d", "p", "q"):
            raise ValueError("The 'type' argument must be 'd', 'p', or 'q'.")
        variable_mapping = {"d": "x", "p": "q", "q": "p"}
        
        def safe_eval(expr, var):
            # Check if 'x' is present in the expression
            if 'x' in expr:
                local_vars = {variable_mapping[type]: var}
                # Update the expression to replace 'x' with the correct variable
                expr_updated = expr.replace("x", variable_mapping[type])
                return eval(expr_updated, globals(), local_vars)
            else:
                # If 'x' is not present, evaluate the expression as a constant
                return eval(expr, globals())
        
        pdqr_fun = getattr(pdqr, f'{type}{self.name}')
        params = inspect.signature(pdqr_fun).parameters
        args = {}
        if 'mu' in params:
            args['mu'] = self.mu
        if 'sigma' in params:
            args['sigma'] = self.sigma
        if 'skew' in params:
            args['skew'] = self.skew
        if 'shape' in params:
            args['shape'] = self.shape
        if 'lamda' in params:
            args['lamda'] = self.lamda
        combined_function = lambda x: safe_eval(fun_str, x) * pdqr_fun(x, **args)[0]
        result, error = quad(combined_function, lower, upper)
        return result
    
    # alias for expectation to conform with scipy.stats
    def expect(self, fun_str:str = "np.abs(x)", type:str = "d", lower:float = -np.inf, upper:float = np.inf):
        """
        Alias method for expectation
        """
        return self.expectation(fun_str = fun_str, type = type, lower = lower, upper = upper)



@dataclass(kw_only=True)
class EstimatedDistribution(Distribution):
    """
    Estimated Distribution Class

    Generated when calling the `estimate` methodd on a Distribution object.
    """
    parameters:Array
    hessian:Array
    scores:Array
    scaler:Array
    index:Array
    loglikelihood:float
    x:Array
    no_obs:int
    solution:Any
    
    def coef(self)->Dict:
        """
        Distribution Parameters
        """
        par_name = _distribution_par_names(self.name)
        pars = {'mu':self.mu, 'sigma':self.sigma, 'skew':self.skew, 'shape':self.shape, 'lamda':self.lamda}
        selected_pars = {key: pars[key] for key in par_name if key in pars}
        return selected_pars
    
    def loglik(self):
        """
        Log Likelihood
        """
        return self.loglikelihood

    def aic(self):
        """
        Akaike's Information Criterion
        """
        # account for any fixed parameters
        npars = len(self.parameters[self.index])
        return -2.0 * self.loglikelihood + 2. * npars

    def bic(self):
        """
        Baysian Information Criterion
        """
        # account for any fixed parameters
        npars = len(self.parameters[self.index])
        nobs = self.no_obs
        return -2.0 * self.loglikelihood + npars * np.log(nobs)

    def vcov(self, type:str = 'H')->Vector:
        """
        Variance-Covariance Matrix of Parameter Estimates

        :param type: the vcov type. Valid choices are H (Hessian), OPG (outer product of gradients) and QMLE (Quasi Maximum Likelihood)
        :rtype: a numpy vector
        """
        V = vcov(hessian = self.hessian, scores = self.scores, type = type)
        return V


    def summary(self, type:str = 'H', decimals:int = 2, numalign = 'decimal', tablefmt:str = 'pretty'):
        """
        Parameter Estimation Summary

        Provides a printout summary of the parameter estimates.

        :param type: the vcov type. Valid choices are H (Hessian), OPG (outer product of gradients) and QMLE (Quasi Maximum Likelihood)
        :param decimals: number of decimals to print
        :param numalign: number alignment for package tabulate
        :param tablefmt: table format for package tabulate
        :rtype: a console printout of the summary
        """
        # deal with fixed parameter case
        parameter_names = _distribution_par_names(self.name)
        parameters = self.parameters
        V = vcov(hessian = self.hessian, scores = self.scores, type = type)
        n = len(parameter_names)
        index = self.index
        se = np.nan * np.ones(n)
        se[index] = np.sqrt(np.diag(V))
        tval = np.nan * np.ones(n)
        tval[index] = parameters[index]/se[index]
        p = np.nan * np.ones(n)
        p[index] = 2 * (1 - pdqr.pnorm(abs(tval[index])))
        table_data = list(zip(parameter_names, _format_list(parameters.tolist(), decimals), _format_list(se.tolist(), decimals), _format_list(tval.tolist(), decimals), _format_list(p.tolist(),decimals)))
        headers = ["", "Estimate", "Std. Error", "t value","Pr(>|t|)"]
        table = tabulate(table_data, headers=headers, tablefmt=tablefmt, stralign='left', numalign=numalign)
        table_width = max(len(row) for row in table.split("\n"))
        separator_line = '=' * table_width
        print('\n')
        print(f'Distribution:',self.name)
        skewness = pdqr.skewness(distribution = self.name, mu = self.mu, sigma = self.sigma, skew = self.skew, shape = self.shape, lamda = self.lamda)
        kurtosis = pdqr.kurtosis(distribution = self.name, mu = self.mu, sigma = self.sigma, skew = self.skew, shape = self.shape, lamda = self.lamda)
        skewness = _format_list(skewness.tolist(), decimals)[0]
        kurtosis = _format_list(kurtosis.tolist(), decimals)[0]
        AIC = _format_list(np.atleast_1d(self.aic()), decimals)[0]
        BIC = _format_list(np.atleast_1d(self.bic()), decimals)[0]
        left_label = {" No. Observations":self.no_obs, " No. Parameters":len(index), " Skewness":skewness, " Kurtosis":kurtosis," Covariance":type}
        right_label = {"Log Likelihood":round(self.loglikelihood, decimals),"AIC":AIC, "BIC":BIC}
        print_table(left_label, right_label, table_width)
        print(table)
    
    def plot(self, type = 'density')->ggplot:
        """
        Estimated Distribution PDF Plot
        
        :param type: the type of plot. Valid choices are 'density' and 'qq'
        :rtype: a ggplot
        """
        dfun = f'pdqr.d{self.name}'
        qfun = f'pdqr.q{self.name}'
        parameter_names = _distribution_par_names(self.name)
        parameters = self.parameters
        params = dict(zip(parameter_names, parameters))
        if type == 'density':
            return pdfplot(self.x, eval(dfun), **params)
        else:
            return qqplot(self.x, eval(qfun), **params) + qqline(self.x, eval(qfun), **params)
        


@dataclass(kw_only=True)
class ProfiledDistribution(Distribution):
    """
    Profiled Distribution Class

    Generated when calling the `profile` methodd on a Distribution object.

    """
    dist:Dict[int, np.ndarray]
    rmse:Dict[int, np.ndarray]
    sim:int
    size:list

    def summary(self, numalign = 'center', floatfmt=".2f", tablefmt:str = 'psql'):
        """
        Profile Distribution Summary

        Provides a printout summary of the profile distribution results.

        :param numalign: number alignment for package tabulate
        :param floatfmt: float format for package tabulate
        :param tablefmt: table format for package tabulate
        :rtype: a console printout of the summary
        """
        summary_data = []
        parameter_names = _distribution_par_names(self.name)
        pars = {'mu':self.mu, 'sigma':self.sigma, 'skew':self.skew, 'shape':self.shape, 'lamda':self.lamda}
        selected_pars = {key: round(pars[key],3) for key in parameter_names if key in pars}

        # Iterate through each size in results_dict
        for size, simulations in self.dist.items():
            num_parameters = simulations.shape[1]  # Assuming second dimension is number of parameters  
            for param_index in range(num_parameters):
                param_name = parameter_names[param_index]
                mean_val = np.mean(simulations[:, param_index])
                min_val = np.min(simulations[:, param_index])
                max_val = np.max(simulations[:, param_index])
                rmse_val = self.rmse[size][param_index]
                summary_data.append([size, param_name, mean_val, min_val, max_val, rmse_val])
        # include actual values, distribution and sims in output printout
        summary_df = pd.DataFrame(summary_data, columns=['Size', 'Parameter', 'Mean', 'Min', 'Max', 'RMSE'])
        summary_df['Parameter'] = pd.Categorical(summary_df.Parameter, ordered=True, categories=parameter_names)
        summary_df = summary_df.sort_values(by = 'Parameter')
        summary_df['Actual'] = summary_df['Parameter'].map(selected_pars)
        column_order = ['Size', 'Parameter', 'Actual', 'Mean', 'Min', 'Max', 'RMSE']
        summary_df = summary_df[column_order]
        print('\n')
        print(f'Distribution:',self.name)
        print(f'Simulations/Size:',self.sim)
        print(tabulate(summary_df.values.tolist(), headers=summary_df.columns.tolist(), showindex=False, floatfmt = floatfmt, stralign = 'left', numalign = numalign, tablefmt = tablefmt))


    def pandas(self, type:str = 'wide')->pd.DataFrame:
        """
        Profiled Distribution to Pandas DataFrame

        :param type: the type of formatted output. Valid choices are 'wide' and 'long'
        :rtype: a pandas DataFrame
        """
        pandas_df = pd.DataFrame()
        for size, data_array in self.dist.items():
            num_sims = data_array.shape[0]  # Number of simulations
            # Create a DataFrame from the NumPy array
            temp_df = pd.DataFrame(data_array, columns=['mu', 'sigma', 'skew', 'shape'])
            # Add columns for the size and simulation number
            temp_df['size'] = size
            temp_df['sim'] = np.arange(num_sims) + 1
            # Concatenate this DataFrame with the final DataFrame
            pandas_df = pd.concat([pandas_df, temp_df], ignore_index=True)
        column_order = ['size', 'sim'] + [col for col in pandas_df.columns if col not in ['size', 'sim']]
        pandas_df = pandas_df[column_order]

        if type == 'long':
            # Melt the DataFrame to long format
            pandas_df = pandas_df.melt(id_vars=['size', 'sim'], var_name='parameter', value_name='value')

        return pandas_df


    def plot(self, parameter:str = 'mu')->ggplot:
        """
        Profiled Distribution MSE Box Plot

        :param parameter: the parameter to plot. Valid choices are 'mu', 'sigma', 'skew', 'shape' and 'lambda'
        :rtype: a ggplot
        """
        if parameter not in ['mu', 'sigma', 'skew', 'shape', 'lamda']:
            raise ValueError("The 'parameter' argument must be 'mu', 'sigma', 'skew', 'shape' or 'lamda'.")
        # check is self.name has the parameters
        parameter_names = _distribution_par_names(self.name)
        if parameter not in parameter_names:
            raise ValueError(f"The '{self.name}' distribution does not have a '{parameter}' parameter.")
        # Create a DataFrame from the NumPy array
        pandas_df = self.pandas(type='long')
        # Filter the DataFrame to the specified parameter
        pandas_df = pandas_df[pandas_df.parameter == parameter]
        # Create a ggplot object
        p = (
            ggplot(pandas_df, aes(x = 'factor(size)', y = 'value')) +
            geom_boxplot() +
            labs(x='Sample Size', y='MSE')
        )
        return p


    # overload pqdr and include size and sim as inputs to randomly select parameters from the distribution
    def cdf(self, q: Vector, sim:int | None = None, size:int | None = None, lower_tail:bool = True) -> Array:
        """
        Overridden method to calculate a modified Cumulative Probability Function specific to ProfiledDistribution.

        The distribution parameters are selected randomly from the profile distribution if they are not specified,
        else can be selected based on their index (sim and size).

        :param q: a vector of quantiles
        :param sim: the simulation number
        :param size: the sample size
        :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
        :rtype: a numpy array
        """
        if sim is not None and (sim < 1 or sim > self.sim):
            raise ValueError(f"sim must be between 1 and {self.sim}, inclusive")

        if size is not None and size not in self.size:
            raise ValueError(f"size must be one of the following values: {self.size}")

        actual_sim: int = np.random.randint(0, self.sim - 1) if sim is None else (sim - 1)
        actual_size: int = np.random.choice(self.size) if size is None else size
        parameters = self.dist[actual_size][actual_sim, :]
        parameter_names = _distribution_par_names(self.name)
        params = dict(zip(parameter_names, parameters))
        return super().cdf(q, **params, lower_tail = lower_tail)
    
    def pdf(self, x: Vector, sim:int | None = None, size:int | None = None, log:bool = False) -> Array:
        """
        Overridden method to calculate a modified Probability Density Function specific to ProfiledDistribution.

        The distribution parameters are selected randomly from the profile distribution if they are not specified,
        else can be selected based on their index (sim and size).

        :param x: a vector of quantiles
        :param sim: the simulation number
        :param size: the sample size
        :param log: whether to return the log density
        :rtype: a numpy array
        """
        if sim is not None and (sim < 1 or sim > self.sim):
            raise ValueError(f"sim must be between 1 and {self.sim}, inclusive")

        if size is not None and size not in self.size:
            raise ValueError(f"size must be one of the following values: {self.size}")

        actual_sim: int = np.random.randint(0, self.sim - 1) if sim is None else (sim - 1)
        actual_size: int = np.random.choice(self.size) if size is None else size
        parameters = self.dist[actual_size][actual_sim, :]
        parameter_names = _distribution_par_names(self.name)
        params = dict(zip(parameter_names, parameters))
        return super().pdf(x, **params, log = log)

    def quantile(self, p: Vector, sim:int | None = None, size:int | None = None, lower_tail:bool = True) -> Array:
        """
        Overridden method to calculate a modified Probability Density Function specific to ProfiledDistribution.

        The distribution parameters are selected randomly from the profile distribution if they are not specified,
        else can be selected based on their index (sim and size).

        :param p: a vector of probabilities
        :param sim: the simulation number
        :param size: the sample size
        :param lower_tail: if TRUE (default), probabilities are P[X ≤ x] otherwise, P[X > x]
        :rtype: a numpy array
        """
        if sim is not None and (sim < 1 or sim > self.sim):
            raise ValueError(f"sim must be between 1 and {self.sim}, inclusive")

        if size is not None and size not in self.size:
            raise ValueError(f"size must be one of the following values: {self.size}")

        actual_sim: int = np.random.randint(0, self.sim - 1) if sim is None else (sim - 1)
        actual_size: int = np.random.choice(self.size) if size is None else size
        parameters = self.dist[actual_size][actual_sim, :]
        parameter_names = _distribution_par_names(self.name)
        params = dict(zip(parameter_names, parameters))
        return super().quantile(p, **params, lower_tail = lower_tail)

    def ppf(self, p: Vector, sim:int | None = None, size:int | None = None, lower_tail:bool = True) -> Array:
        """
        Alias method for quantile function
        """
        return self.quantile(p = p, sim = sim, size = size, lower_tail = lower_tail)
    

    def random(self, n: int = 1, sim:int | None = None, size:int | None = None, seed: Optional[int] = None) -> Array:
        """
        Overridden method to generate a modified random sample specific to ProfiledDistribution.

        The distribution parameters are selected randomly from the profile distribution if they are not specified,
        else can be selected based on their index (sim and size).

        :param n: the number of draws
        :param sim: the simulation number
        :param size: the sample size
        :param seed: an optional value to initialize the random seed generator
        :rtype: a numpy array
        """
        if sim is not None and (sim < 1 or sim > self.sim):
            raise ValueError(f"sim must be between 1 and {self.sim}, inclusive")

        if size is not None and size not in self.size:
            raise ValueError(f"size must be one of the following values: {self.size}")

        actual_sim: int = np.random.randint(0, self.sim - 1) if sim is None else (sim - 1)
        actual_size: int = np.random.choice(self.size) if size is None else size
        parameters = self.dist[actual_size][actual_sim, :]
        parameter_names = _distribution_par_names(self.name)
        params = dict(zip(parameter_names, parameters))
        return super().random(n, **params, seed = seed)
    
    def rvs(self, n: int = 1, sim:int | None = None, size:int | None = None, seed: Optional[int] = None) -> Array:
        """
        Alias method for random function
        """
        return self.random(n = n, sim = sim, size = size, seed = seed)
    

