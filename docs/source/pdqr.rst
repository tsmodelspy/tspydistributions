pdqr module
===========

All distributions are parameterized by the mean, standard deviation (sigma), skew and shape parameters. For the Generalized Hyperbolic
distribution, there is an additional parameter `lamda`, representing a real parameter in the Generalized Inverse Gaussian (`GIG`) 
distribution of the Normal mean variance mixture representation of the distribution.
For each distribution (`*`), there is a function for the cumulative distribution (`p*`), density (`d*`), 
quantile (`q*`) and random sampling (`r*`). Additional functions for calculating the skewness and kurtosis given the 
skew and shape parameters are also available.

Normal distribution
-------------------
.. include:: normal.md
   :parser: myst_parser.sphinx_

.. autofunction:: pdqr.dnorm
.. autofunction:: pdqr.pnorm
.. autofunction:: pdqr.qnorm
.. autofunction:: pdqr.rnorm


Student's T distribution
------------------------
.. include:: student.md
   :parser: myst_parser.sphinx_
.. autofunction:: pdqr.dstd
.. autofunction:: pdqr.pstd
.. autofunction:: pdqr.qstd
.. autofunction:: pdqr.rstd



Generalized Error distribution
------------------------------
.. include:: ged.md
   :parser: myst_parser.sphinx_
.. autofunction:: pdqr.dged
.. autofunction:: pdqr.pged
.. autofunction:: pdqr.qged
.. autofunction:: pdqr.rged

Skewed Distributions by Inverse Scale Factors
---------------------------------------------
.. include:: skewed_fernandez_steel.md
   :parser: myst_parser.sphinx_

Skew Normal distribution
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: pdqr.dsnorm
.. autofunction:: pdqr.psnorm
.. autofunction:: pdqr.qsnorm
.. autofunction:: pdqr.rsnorm

Skew Generalized Error distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: pdqr.dsged
.. autofunction:: pdqr.psged
.. autofunction:: pdqr.qsged
.. autofunction:: pdqr.rsged

Skew Student's T distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: pdqr.dsstd
.. autofunction:: pdqr.psstd
.. autofunction:: pdqr.qsstd
.. autofunction:: pdqr.rsstd

Johnson's SU distribution
-------------------------
.. include:: johnsonsu.md
   :parser: myst_parser.sphinx_
.. autofunction:: pdqr.djsu
.. autofunction:: pdqr.pjsu
.. autofunction:: pdqr.qjsu
.. autofunction:: pdqr.rjsu

Generalized Hyperbolic distribution
-----------------------------------
.. include:: ghyp.md
   :parser: myst_parser.sphinx_
.. autofunction:: pdqr.dsgh
.. autofunction:: pdqr.psgh
.. autofunction:: pdqr.qsgh
.. autofunction:: pdqr.rsgh

Generalized Hyperbolic Skew Student's T distribution
----------------------------------------------------
.. include:: ghst.md
   :parser: myst_parser.sphinx_
.. autofunction:: pdqr.dsghst
.. autofunction:: pdqr.psghst
.. autofunction:: pdqr.qsghst
.. autofunction:: pdqr.rsghst
