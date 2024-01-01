In distributions where the expected moments are functions of all the parameters, it is not immediately obvious how to perform such a transformation. In the case of the Generalized Hyperbolic (GH) distribution, because of the existence of location and scale invariant parameterizations and the possibility of expressing 
the variance in terms of one of those, namely the $(\zeta, \rho)$, the task of standardizing and estimating the density can be broken down to one of estimating those 2 parameters, representing a combination of shape and skewness, followed by a series of transformation steps to demean, scale and then translate the parameters 
into the $(\alpha, \beta, \delta, \mu)$ parameterization for which standard formula exist for the likelihood function. The $(\xi, \chi)$ parameterization, which is a simple transformation of the $(\zeta, \rho)$, could also be used in the first step and then transformed into the latter before proceeding further. The only difference is the kind of 'immediate' inference one can make from the different parameterizations, each providing a different direct insight into the kind of dynamics produced and their place in the overall GH family particularly with regards to the limit cases.

The package performs estimation using the $(\zeta, \rho)$ parameterization, after which a series of steps transform those parameters into the $(\alpha, \beta, \delta, \mu)$ while at the same time including the necessary recursive substitution of parameters in order to standardize the resulting distribution.


Consider the standardized Generalized Hyperbolic Distribution. Let $\varepsilon_t$ be a r.v. with mean $(0)$ and variance $({\sigma}^2)$ distributed as $\textrm{GH}(\zeta, \rho)$, and let $z$ be a scaled version of the r.v. $\varepsilon$ with variance $(1)$ and also distributed as $\textrm{GH}(\zeta, \rho)$ (the parameters
$\zeta$ and $\rho$ do not change as a result of being location and scale invariant). The density $f(.)$ of $z$ can be expressed as

$$
f(\frac{\varepsilon_t}{\sigma}; \zeta ,\rho ) = \frac{1}{\sigma}f_t(z;\zeta ,\rho ) =
\frac{1}{\sigma}f_t(z;\tilde \alpha, \tilde \beta, \tilde \delta ,\tilde \mu ),
$$

where we make use of the $(\alpha, \beta, \delta, \mu)$ parameterization since we can only naturally express the density in that parameterization. The steps to
transforming from the $(\zeta, \rho)$ to the $(\alpha, \beta, \delta, \mu)$ parameterization, while at the same time standardizing for zero mean and unit
variance are given henceforth. Let

$$
\begin{eqnarray}
\zeta & = & \delta \sqrt {{\alpha ^2} - {\beta ^2}} \hfill \\
\rho & = & \frac{\beta }{\alpha }, \hfill
\end{eqnarray}
$$

which after some substitution may be also written in terms of  $\alpha$ and $\beta$ as,

$$
\begin{eqnarray}
\alpha & = & \frac{\zeta }{{\delta \sqrt {(1 - {\rho ^2})} }},\hfill\\
\beta  & = &\alpha \rho.\hfill
\end{eqnarray}
$$

For standardization we require that,

$$
\begin{eqnarray}
  E\left(X\right) & = & \mu  + \frac{{\beta \delta }}{{\sqrt {{\alpha ^2} - {\beta ^2}} }}\frac{{{K_{\lambda  + 1}}\left(\zeta \right)}}{{{K_\lambda }\left(\zeta \right)}} = \mu  + \frac{{\beta {\delta ^2}}}{\zeta }\frac{{{K_{\lambda  + 1}}\left(\zeta \right)}}{{{K_\lambda }\left(\zeta \right)}} = 0 \hfill \\
  \therefore \mu  & = & - \frac{{\beta {\delta ^2}}}{\zeta }\frac{{{K_{\lambda  + 1}}\left(\zeta \right)}}{{{K_\lambda }\left(\zeta \right)}}\hfill \\
  Var\left(X\right) & = & {\delta ^2}\left(\frac{{{K_{\lambda  + 1}}\left(\zeta \right)}}{{\zeta {K_\lambda }\left(\zeta \right)}} + \frac{{{\beta ^2}}}{{{\alpha ^2} - {\beta ^2}}}\left(\frac{{{K_{\lambda  + 2}}\left(\zeta \right)}}{{{K_\lambda }\left(\zeta \right)}} - {\left(\frac{{{K_{\lambda  + 1}}\left(\zeta \right)}}{{{K_\lambda }\left(\zeta \right)}}\right)^2}\right)\right) = 1 \hfill\nonumber \\
  \therefore \delta  & = & {\left(\frac{{{K_{\lambda  + 1}}\left(\zeta \right)}}{{\zeta {K_\lambda }\left(\zeta \right)}} + \frac{{{\beta ^2}}}{{{\alpha ^2} - {\beta ^2}}}\left(\frac{{{K_{\lambda  + 2}}\left(\zeta \right)}}{{{K_\lambda }\left(\zeta \right)}} - {\left(\frac{{{K_{\lambda  + 1}}\left(\zeta \right)}}{{{K_\lambda }\left(\zeta \right)}}\right)^2}\right)\right)^{ - 0.5}} \hfill
\end{eqnarray}
$$


Since we can express, $\beta^2/\left(\alpha^2 - \beta^2\right)$ as,

$$
\frac{{{\beta ^2}}}{{{\alpha ^2} - {\beta ^2}}} = \frac{{{\alpha ^2}{\rho ^2}}}{{{a^2} - {\alpha ^2}{\rho ^2}}} = \frac{{{\alpha ^2}{\rho ^2}}}{{{a^2}\left(1 - {\rho ^2}\right)}} = \frac{{{\rho ^2}}}{{\left(1 - {\rho ^2}\right)}},
$$

then we can re-write the formula for $\delta$ in terms of the estimated parameters $\hat\zeta$ and $\hat\rho$ as,

$$
\delta  = {\left(\frac{{{K_{\lambda  + 1}}\left(\hat \zeta \right)}}{{\hat \zeta {K_\lambda }\left(\hat \zeta \right)}} + \frac{{{{\hat \rho }^2}}}{{\left(1 - {{\hat \rho }^2}\right)}}\left(\frac{{{K_{\lambda  + 2}}\left(\hat \zeta \right)}}{{{K_\lambda }\left(\hat \zeta \right)}} - {\left(\frac{{{K_{\lambda  + 1}}\left(\hat \zeta \right)}}{{{K_\lambda }\left(\hat \zeta \right)}}\right)^2}\right)\right)^{ - 0.5}}
$$

Transforming into the $(\tilde \alpha ,\tilde \beta ,\tilde \delta ,\tilde \mu )$
parameterization proceeds by first substituting the above value of $\delta$ into the
equation for $\alpha$ and simplifying:

$$
\begin{eqnarray}
  \tilde \alpha & = & \,{\frac{{\hat \zeta \left( {\frac{{{{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)}}{{\hat \zeta {{\text{K}}_\lambda }\left( {\hat \zeta } \right)}} + \frac{{{{\hat \rho }^2}\left( {\frac{{{{\text{K}}_{\lambda  + 2}}\left( {\hat \zeta } \right)}}{{{{\text{K}}_\lambda }\left( {\hat \zeta } \right)}} - \frac{{{{\left( {{{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)} \right)}^2}}}{{{{\left( {{{\text{K}}_\lambda }\left( {\hat \zeta } \right)} \right)}^2}}}} \right)}}{{\left( {1 - {{\hat \rho }^2}} \right)}}} \right)}}{{\sqrt {(1 - {{\hat \rho }^2})} }}^{0.5}}, \hfill\nonumber \\
   & = &\,{\frac{{\left( {\frac{{\hat \zeta {{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)}}{{{{\text{K}}_\lambda }\left( {\hat \zeta } \right)}} + \frac{{{{\hat \zeta }^2}{{\hat \rho }^2}\left( {\frac{{{{\text{K}}_{\lambda  + 2}}\left( {\hat \zeta } \right)}}{{{{\text{K}}_\lambda }\left( {\hat \zeta } \right)}} - \frac{{{{\left( {{{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)} \right)}^2}}}{{{{\left( {{{\text{K}}_\lambda }\left( {\hat \zeta } \right)} \right)}^2}}}} \right)}}{{\left( {1 - {{\hat \rho }^2}} \right)}}} \right)}}{{\sqrt {(1 - {\hat \rho ^2})} }}^{0.5}}, \hfill\nonumber \\
   & = & {\left( {\left. {\frac{{\frac{{\hat \zeta {{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)}}{{{{\text{K}}_\lambda }\left( {\hat \zeta } \right)}}}}{{(1 - {{\hat \rho }^2})}} + \frac{{{\hat \zeta ^2}{\hat \rho ^2}\left( {\frac{{{{\text{K}}_{\lambda  + 2}}\left( {\hat \zeta } \right)}}{{{{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)}}\frac{{{{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)}}{{{{\text{K}}_\lambda }\left( {\hat \zeta } \right)}} - \frac{{{{\left( {{{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)} \right)}^2}}}{{{{\left( {{{\text{K}}_\lambda }\left( {\hat \zeta } \right)} \right)}^2}}}} \right)}}{{{{\left( {1 - {{\hat \rho }^2}} \right)}^2}}}} \right)} \right.^{0.5}}, \hfill\nonumber \\
   & = & {\left( {\left. {\frac{{\frac{{\hat \zeta {{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)}}{{{{\text{K}}_\lambda }\left( {\hat \zeta } \right)}}}}{{(1 - {{\hat \rho }^2})}}\left(1 + \frac{{\hat \zeta {{\hat \rho }^2}\left( {\frac{{{{\text{K}}_{\lambda  + 2}}\left( {\hat \zeta } \right)}}{{{{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)}} - \frac{{{{\text{K}}_{\lambda  + 1}}\left( {\hat \zeta } \right)}}{{{{\text{K}}_\lambda }\left( {\hat \zeta } \right)}}} \right)}}{{\left( {1 - {{\hat \rho }^2}} \right)}}\right)} \right)} \right.^{0.5}}. \hfill
\end{eqnarray}
$$

Finally, the rest of the parameters are derived recursively from $\tilde\alpha$
and the previous results,

$$
\begin{eqnarray}
  \tilde \beta  & = & \tilde \alpha \hat \rho,\hfill\\
  \tilde \delta & = & \frac{{\hat \zeta }}{{\tilde \alpha \sqrt {1 - {{\hat \rho }^2}} }}, \hfill\\
  \tilde \mu & = & \frac{{ - \tilde \beta {{\tilde \delta }^2}{K_{\lambda  + 1}}\left(\hat \zeta \right)}}{{\hat \zeta {K_\lambda }\left(\hat \zeta \right)}}.\hfill
\end{eqnarray}
$$

For the use of the $(\xi, \chi)$ parameterization in estimation, the additional
preliminary steps of converting to the $(\zeta, \rho)$ are,

$$
\begin{eqnarray}
  \zeta  & = & \frac{1}{{{{\hat \xi }^2}}} - 1, \hfill\\
  \rho  & = & \frac{{\hat \chi }}{{\hat \xi }}. \hfill
\end{eqnarray}
$$
