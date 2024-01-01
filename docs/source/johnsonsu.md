In the original parameterization of Johnson's SU distribution, the pdf is given by:

$$
f\left(x, \xi, \lambda, \nu^*, \tau^*\right) = \frac{\tau^*}{\sigma\left(s^2 + 1\right)^{0.5}\sqrt{2\pi}}\exp\left[-\frac{1}{2}z^2\right]
$$

for $-\infty<x<\infty$, $-\infty<\xi<\infty$, $\lambda>0$, $-\infty<\nu^*<\infty$ and $\tau^*>0$, where 

$$
z = \nu^* + \tau^* \sinh^{-1}\left(s\right) = \nu^* + \tau^*\log\left[s + \left(s^2 + 1\right)^{0.5}\right]
$$

with $s = \left(x - \xi)/\lambda\right)$.

Reparameterizing the pdf in terms of the mean and standard deviation we set $\mu = \xi - \lambda\omega^{0.5}\sinh\left(\nu^*/\tau^*\right)$,
$\sigma = \lambda/c$, $\nu = -\nu^*$ and $\tau = \tau^*$. Therefore:

$$
f\left(x, \mu, \sigma, \nu, \tau\right) = \frac{\tau}{c\sigma\left(s^2+1\right)^{0.5}\sqrt(2\pi)}\exp\left[-\frac{1}{2}z^2\right]
$$

for $-\infty<x<\infty$, $-\infty<\mu<\infty$, $\sigma>0$, $-\infty<\nu<\infty$ and $\tau>0$, where 

$$
\begin{aligned}
z &= -\nu + \tau\sinh^{-1}\left(s\right) = -\nu + \tau\log\left(s + \left(s^2 + 1\right)^{0.5}\right) \hfill \\
s &= \frac{x - \mu + c\sigma\omega^{0.5}\sinh\left(\nu/\tau\right)}{c\sigma} \hfill \\
c &=  \left\{0.5\left(\omega - 1\right)\left[\omega\cosh\left(2\nu/\tau\right) + 1\right]\right\}^{-0.5} \hfill \\
\omega &= \exp\left(1/\tau^2\right)
\end{aligned}
$$

The reparameterization is taken from {cite:p}`Rigby2019` Section 18.4.3 which also contains additional information on the
properties of the $\nu$ and $\tau$ parameters.
