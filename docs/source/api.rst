api module
===========
With the `Distribution` class, one can :

* define a distribution with a fixed set of parameters,
* estimate parameters given a sample (`estimate`, alias `fit`),
* calculate the `pdf`, `cdf`, `quantile` (alias `ppf``) or generate `random`` samples (alias `rvs``),
* profile the distribution parameters through a simulation study (`profile`),
* evaluate expectations on the distibution using quadrature (`expectation`, alias `expect`),
* calculate the `skewness` and `kurtosis` of the distribution.

The `estimate` method outputs a subclass of `Distribution` called `EstimatedDistribution` which inherits the methods from 
`Distribution` in addition to a `summary`, `vcov`, `bic`, `aic`, `loglik`` and a `plot`` method. 

The `profile` method outputs a subclass of `Distribution` called `ProfiledDistribution` which inherits the methods from 
`Distribution` to `summary`, conversion to a `pandas` dataframe and `plot`` methods for the Mean Squared Error (MSE) of 
the estimated parameters across different sample sizes. Additionally, an adjusted set of methods for `pdf`, `cdf`, `quantile` 
and `random` are available which make use of the profiled parameters instead of the estimated parameters (or randomly selected 
if none provided), allowing one to evaluate the distribution on the profiled parameters based on random draws from different sample sizes.

.. autoclass:: api.Distribution(name = 'norm', mu = 0, sigma = 1, skew = 0, shape = 5, lamda = 0)
   :members:
   :exclude-members: _name, _sigma
   :member-order: bysource

.. autoclass:: api.EstimatedDistribution
   :members:
   :exclude-members: _name, _sigma
   :member-order: bysource

.. autoclass:: api.ProfiledDistribution
   :members:
   :exclude-members: _name, _sigma
   :member-order: bysource