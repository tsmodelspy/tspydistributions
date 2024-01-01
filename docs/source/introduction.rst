Welcome to tspydistributions
============================

The **tspydistributions** package implements a number of location-scale based statistical
distributions parameterized in terms of their mean and standard deviation, in addition to the
skew and shape parameters. The functions in the pdqr module follow R's distributions naming 
convention in terms of pdf (d), cdf (p), quantile (q) and random number generation (r). 
Where possible, the functions are vectorized and can be used with numpy arrays.

The api module allows the creation of a `Distribution` object, which can be used to
estimate and profile the parameters of the distribution. The estimation is carried out
using pytorch and the scipy minimize function. For the Generalized Hyperbolic and Generalized 
Hyperbolic Skew Student distributions, the estimation is carried out using numerical derivatives 
until such time as the modified Bessel function of the second kind is implemented in pytorch. 
Some aliases similar to scipy.stats are provided.

Currently implemented distributions are the Normal ('norm'), Student ('std') and Generalized 
Error ('ged') distributions; the skewed variants of these based on the transformations 
in :cite:p:`Fernandez1998`, which are the Skew Normal ('snorm'), Skew Student ('sstd') and 
Skew Generalized Error ('sged`) distributions; The reparameterized version of :cite:p:`Johnson1949` 
SU distribution ('jsu'); the Generalized Hyperbolic ('sgh') distribution of :cite:p:`Barndorff1977` 
and the Generalized Hyperbolic Skew Student ('sghst') distribution of :cite:p:`Aas2006`

This is a python implementation of the R package with the same name and by the same author.
The github repository is located `here <https://github.com/tsmodelspy/tspydistributions>`_
