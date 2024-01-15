density module
====================

This module contains torch implementations of the density functions which can be
used in combination with torch based models wanting to make use of the distributions
in this package. Unlike the pdqr module which is based on numpy, this module can take
care of the gradients for you.

Note that the Generalized Hyperbolic and Generalized Hyperbolic Skew Student's Distributions
will not work with autograd, since the modified bessel function of the second kind is not yet
implemented in torch.

An example is available in the the tspydistributions gallery.

Density Functions
-----------------

.. autofunction:: density.dnorm
.. autofunction:: density.dged
.. autofunction:: density.dstd
.. autofunction:: density.dsnorm
.. autofunction:: density.dsged
.. autofunction:: density.dsstd
.. autofunction:: density.djsu
.. autofunction:: density.dsgh
.. autofunction:: density.dsghst
