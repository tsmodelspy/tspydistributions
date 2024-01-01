{cite:p}`Fernandez1998` proposed introducing skewness into unimodal and symmetric
distributions by introducing inverse scale factors in the positive and negative
real half lines.

Given a skew parameter, $\xi$ (when $\xi=1$, the distribution is symmetric), 
the density of a random variable z can be represented as:

$$
\begin{equation}
f\left( {z|\xi } \right) = \frac{2}
{{\xi  + {\xi ^{ - 1}}}}\left[ {f\left( {\xi z} \right)H\left( { - z} \right) + f\left( {{\xi ^{ - 1}}z} \right)H\left( z \right)} \right]
\end{equation}
$$
    
where $\xi  \in {\mathbb{R}^ + }$ and $H(.)$ is the Heaviside function. The
absolute moments, required for deriving the central moments, are generated from
the following function:

$$
\begin{equation}
{M_r} = 2\int_0^\infty  {{z^r}f\left( z \right)dz}.
\end{equation}
$$

The mean and variance are then defined as:

$$
\begin{gathered}
  E\left( z \right) = {M_1}\left( {\xi  - {\xi ^{ - 1}}} \right) \hfill \\
  Var\left( z \right) = \left( {{M_2} - M_1^2} \right)\left( {{\xi ^2} + {\xi ^{ - 2}}} \right) + 2M_1^2 - {M_2} \hfill \\
\end{gathered}
$$

The Normal, Student and GED distributions have skew variants which have been
standardized to zero mean, unit variance by making use of the moment conditions
given above.