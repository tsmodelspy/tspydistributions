The GH Skew-Student distribution was popularized by {cite:p}`Aas2006` because of its uniqueness
in the GH family in having one tail with polynomial and one with exponential behavior.
This distribution is a limiting case of the GH when $\alpha  \to \left| \beta  \right|$
and $\lambda=-\nu/2$, where $\nu$ is the shape parameter of the Student distribution.
The domain of variation of the parameters is $\beta  \in \mathbb{R}$ and $\nu>0$, but
for the variance to be finite $\nu>4$, while for the existence of skewness and kurtosis,
$\nu>6$ and $\nu>8$ respectively. The density of the random variable $x$ is then given
by:

$$
f\left( x \right) = \frac{{{2^{\left( {1 - \nu } \right)/2}}{\delta ^\nu }{{\left| \beta  \right|}^{\left( {\nu  + 1} \right)/2}}{K_{\left( {\nu  + 1} \right)/2}}\left( {\sqrt {{\beta ^2}\left( {{\delta ^2} + {{\left( {x - \mu } \right)}^2}} \right)} } \right)\exp \left( {\beta \left( {x - \mu } \right)} \right)}}
{{\Gamma \left( {\nu /2} \right)\sqrt \pi  {{\left( {\sqrt {{\delta ^2} + {{\left( {x - \mu } \right)}^2}} } \right)}^{\left( {\nu  + 1} \right)/2}}}}
$$

To standardize the distribution to have zero mean and unit variance, I make use
of the first two moment conditions for the distribution which are:

$$
\begin{gathered}
  E\left( x \right) = \mu  + \frac{{\beta {\delta ^2}}}
{{\nu  - 2}} \hfill \\
  Var\left( x \right) = \frac{{2{\beta ^2}{\delta ^4}}}
{{{{\left( {\nu  - 2} \right)}^2}\left( {\nu  - 4} \right)}} + \frac{{{\delta ^2}}}
{{\nu  - 2}} \hfill \\
\end{gathered}
$$

We require that $Var(x)=1$, thus:

$$
\delta  = {\left( {\frac{{2{{\bar \beta }^2}}}{{{{\left( {\nu  - 2} \right)}^2}\left( {\nu  - 4} \right)}} + \frac{1}{{\nu  - 2}}} \right)^{ - 1/2}}
$$

where I have made use of the $4^{th}$ parameterization of the GH distribution given in
{cite:p}`Prause1999` where $\hat \beta = \beta \delta$. The location parameter is then rescaled
by substituting into the first moment formula $\delta$ so that it has zero mean:

$$
\bar \mu  =  - \frac{{\beta {\delta ^2}}}{{\nu  - 2}}
$$

Therefore, we model the GH Skew-Student using the location-scale invariant parameterization $(\bar \beta, \nu)$
and then translate the parameters into the usual GH distribution's $(\alpha, \beta, \delta, \mu)$, setting
$\alpha = abs(\beta)+1e-12$.