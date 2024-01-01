For any random variable Z whose probability distribution function belongs to a location scale family, 
the distribution function of $X\stackrel{d}{=} \mu + \sigma Z$ also belongs to the family.

This implies that if Z has a continuous distribution with probability density function (pdf) g, then X 
also has a continuous distribution, with probability density function f given by:

$$
\begin{equation}
f\left(x\right) = \frac{1}{\sigma}g\left(\frac{x-\mu}{\sigma}\right),\quad x\in\mathbb{R}
\end{equation}
$$

The parameter (moment) $\sigma$ affects the pdf through a scaling operation which is why the 
standardized distribution $g$ needs to be adjusted for this, whereas the centering by the mean
($\mu$) is simply a translation operation which does not have any affect on the pdf. 

The distributions implemented in this package all belong to this type of family and are additionaly
parameterized in terms of the mean ($\mu$) and standard deviation ($\sigma$), making for easier 
inference in addition to allowing one to concentrate out the first two moments if desired during
estimation. Specific applications of such distributions and parameterizations can be found in
time series analysis in particular, where the conditional mean ($\mu_t$) and/or standard deviation
($\sigma_t$) have their own motion dynamics, such as in ARMA-GARCH models.



