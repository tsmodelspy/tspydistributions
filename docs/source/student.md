The Student's t distribution is described completely by a shape parameter $\nu$, but 
for standardization we proceed by using its 3 parameter representation as follows:

$$
\begin{equation}
f\left( x \right) = \frac{{\Gamma \left( {\frac{{\nu  + 1}}
{2}} \right)}}
{{\sqrt {\beta \nu \pi } \Gamma \left( {\frac{\nu }
{2}} \right)}}{\left( {1 + \frac{{{{\left( {x - \alpha } \right)}^2}}}
{{\beta \nu }}} \right)^{ - \left( {\frac{{\nu  + 1}}
{2}} \right)}}
\end{equation}
$$

where $\alpha$, $\beta$, and $\nu$ are the location, scale and shape parameters respectively, and $\Gamma$ is the Gamma function. Similar to
the GED distribution described later, this is a unimodal and symmetric distribution
where the location parameter $\alpha$ is the mean (and mode) of the distribution
while the variance is:

$$
\begin{equation}
Var\left( x \right) = \frac{{\beta \nu }}{{\left( {\nu  - 2} \right)}}.
\end{equation}
$$

For the purposes of standardization we require that:

$$
\begin{gathered}
  Var(x) = \frac{{\beta \nu }}
{{\left( {\nu  - 2} \right)}} = 1 \hfill \\
  \therefore \beta  = \frac{{\nu  - 2}}
{\nu } \hfill \\
\end{gathered}
$$

Substituting $\frac{(\nu- 2)}{\nu }$ we obtain the standardized Student's distribution:

$$
\begin{equation}
f\left( {\frac{{x - \mu }}{\sigma }} \right) = \frac{1}
{\sigma }f\left( z \right) = \frac{1}
{\sigma }\frac{{\Gamma \left( {\frac{{\nu  + 1}}
{2}} \right)}}{{\sqrt {\left( {\nu  - 2} \right)\pi } \Gamma \left( {\frac{\nu }
{2}} \right)}}{\left( {1 + \frac{{{z^2}}}
{{\left( {\nu  - 2} \right)}}} \right)^{ - \left( {\frac{{\nu  + 1}}
{2}} \right)}}.
\end{equation}
$$

The Student distribution has zero skewness and excess kurtosis equal to
$6/(\nu  - 4)$ for $\nu > 4$.
