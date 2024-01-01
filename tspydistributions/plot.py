import numpy.typing as npt
import numpy as np
import pandas as pd
from typing import List, TYPE_CHECKING, TypeVar, Callable, Dict
from scipy.stats.mstats import plotting_positions
from plotnine import (
    ggplot,
    geom_point,
    aes,
    stat_function,
    geom_histogram,
    ggtitle,
    theme,
    geom_abline,
    ylab,
    xlab
)


Array = npt.NDArray[np.float64]

def pdfplot(x:Array, dist:Callable, **params)->ggplot:
    """
    Probability Density Function Plot

    :param x: sample data (continuous)
    :param dist: density (pdf) function
    :param params: parameters for the density function
    :return: ggplot object
    """

    dat = pd.DataFrame({"value":x})
    fun = lambda x: dist(x, **params)[0]
    p = (
        ggplot(dat)
        + geom_histogram(aes(x = 'value', y = 'stat(density)'), binwidth=0.25, fill = 'red', alpha = 0.1)
        + stat_function(aes(x = 'value'),fun = fun)
        + ggtitle('Empirical Histogram vs Fitted Density')
        + theme(legend_position='none')
    )
    return p

def qqplot(x:Array, dist:Callable, **params)->ggplot:
    """
    Quantile-Quantile Plot
    
    :param x: sample data (continuous)
    :param dist: distribution quantile (ppf) function
    :param params: parameters for the distribution quantile function
    :return: ggplot object    
    """
    tmp = pd.DataFrame({"value":x})
    sample = tmp["value"].sort_values().values
    quantiles = plotting_positions(sample, alpha = 1/2, beta = 1/2)
    theoretical = dist(quantiles, **params)
    dat = pd.DataFrame({'quantiles':quantiles, "sample":sample, "theoretical":theoretical})
    p = (
    ggplot(data = dat) + geom_point(aes(x = 'theoretical', y = 'sample'))
    + ggtitle('Empirical vs Fitted Quantiles')
    + theme(legend_position='none')
    + ylab('Sample Quantiles') + xlab('Theoretical Quantiles')
    )
    return p

def qqline(x:Array, dist:Callable, prob:List = [0.25, 0.75], **params)->geom_abline:
    """
    Quantile-Quantile Line

    :param x: sample data (continuous)
    :param dist: distribution quantile (ppf) function
    :param prob: vector of length two, representing probabilities. Corresponding quantile pairs define the line drawn.
    :param params: parameters for the distribution quantile function
    :return: geom_abline object
    """

    if len(prob) != 2:
        raise ValueError("prob must be a list of length 2")
    if any(prob) > 1 or any(prob) < 0:
        raise ValueError("prob must be between 0 and 1")
    q = np.asarray(prob, dtype = np.float64)
    y = np.quantile(x, q = q)
    q = dist(prob, **params)
    slope = np.diff(q)/np.diff(y)
    const = q[0] - slope * y[0]
    p = geom_abline(intercept=const, slope= slope)
    return(p)




