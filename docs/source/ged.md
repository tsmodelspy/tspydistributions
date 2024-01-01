The Generalized Error Distribution is a 3 parameter distribution belonging
to the exponential family with conditional density given by,

$$
f\left( x \right) = \frac{{\kappa {e^{ - 0.5{{\left| {\frac{{x - \alpha }}
{\beta }} \right|}^\kappa }}}}}
{{{2^{1 + {\kappa ^{ - 1}}}}\beta \Gamma \left( {{\kappa ^{ - 1}}} \right)}}
$$

with $\alpha$, $\beta$ and $\kappa$ representing the location, scale and
shape parameters. Since the distribution is symmetric and unimodal the location
parameter is also the mode, median and mean of the distribution (i.e. $\mu$).
By symmetry, all odd moments beyond the mean are zero. The variance and kurtosis
are given by,

$$
\begin{gathered}
  Var\left( x \right) = {\beta ^2}{2^{2/\kappa }}\frac{{\Gamma \left( {3{\kappa ^{ - 1}}} \right)}}
{{\Gamma \left( {{\kappa ^{ - 1}}} \right)}} \hfill \\
  Ku\left( x \right) = \frac{{\Gamma \left( {5{\kappa ^{ - 1}}} \right)\Gamma \left( {{\kappa ^{ - 1}}} \right)}}
{{\Gamma \left( {3{\kappa ^{ - 1}}} \right)\Gamma \left( {3{\kappa ^{ - 1}}} \right)}} \hfill \\
\end{gathered}
$$


As $\kappa$ decreases the density gets flatter and flatter while in the limit as
$\kappa  \to \infty$, the distribution tends towards the uniform. Special cases
are the Normal when $\kappa=2$, the Laplace when $\kappa=1$. Standardization is
simple and involves re-scaling the density to have unit standard deviation:

$$
\begin{gathered}
  Var\left( x \right) = {\beta ^2}{2^{2/\kappa }}\frac{{\Gamma \left( {3{\kappa ^{ - 1}}} \right)}}
{{\Gamma \left( {{\kappa ^{ - 1}}} \right)}} = 1 \hfill \\
  \therefore \beta  = \sqrt {{2^{ - 2/\kappa }}\frac{{\Gamma \left( {{\kappa ^{ - 1}}} \right)}}
{{\Gamma \left( {3{\kappa ^{ - 1}}} \right)}}}  \hfill \\
\end{gathered}
$$

Finally, substituting into the scaled density of $z$:

$$
f\left( {\frac{{x - \mu }}
{\sigma }} \right) = \frac{1}
{\sigma }f\left( z \right) = \frac{1}
{\sigma }\frac{{\kappa {e^{ - 0.5{{\left| {\sqrt {{2^{ - 2/\kappa }}\frac{{\Gamma \left( {{\kappa ^{ - 1}}} \right)}}
{{\Gamma \left( {3{\kappa ^{ - 1}}} \right)}}} z} \right|}^\kappa }}}}}
{{\sqrt {{2^{ - 2/\kappa }}\frac{{\Gamma \left( {{\kappa ^{ - 1}}} \right)}}
{{\Gamma \left( {3{\kappa ^{ - 1}}} \right)}}} {2^{1 + {\kappa ^{ - 1}}}}\Gamma \left( {{\kappa ^{ - 1}}} \right)}}
$$