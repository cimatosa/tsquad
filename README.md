# tsquad - Numeric Integration Using tanh-sinh Variable Transformation

The `tsquad` package provides general purpose integration routines.
For simple integration tasks this package is as good as the standard routines
available e.g. by `scipy`. 
Although, the `tsquad` routines can directly integrate handle **complex functions** 
(along the real axes). 
Most strikingly, the `tsquad` method is particularly suited to efficiently handle 
**singularities** by exploiting the *tanh-sinh* variable transformation 
(also known as *double exponential* approach), as nicely described 
here http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/dhb-tanh-sinh.pdf.
As of that, integrals with lower/upper **bounds at infinity** can be treated
within that scheme.

Furthermore, integrals over **oscillatory functions with an infinite boundary** are
handled very efficiently by extrapolating the partial sums (obtained from integrating
piecewise over half the period) using Shanks' transform (Wynn epsilon algorithm).
This approach (or at least similar) is also available in the [GSL-library](
https://www.gnu.org/software/gsl/doc/html/integration.html#qawf-adaptive-integration-for-fourier-integrals).

For a visualization of great speedup of that extrapolation, see the 
example in `examples_shanks.py`.

# Install

## poetry

Use `poetry` to add `tsquad` to your dependencies with `poetry add tsquad`.

Or edit your `pyproject.toml` file like that
    
    [tool.poetry.dependencies]
    tsquad = "^0.2.0"

and run `poetry install`.

## pip

Run `pip install tsquad`.

## git

You find the source at https://github.com/cimatosa/tsquad.

# Examples

## Singularities

Consider the function `f(x) = 1/x^0.9`. It diverges at `x=0`, however, the integral
over `[0, 1]` is still finite (`I=10`).
Using the tanh-sinh integration scheme allows to efficiently obtained a highly
accurate numerical result.

    >>> import tsquad
    >>> f = lambda x: 1/x**0.9
    >>> tsq = tsquad.QuadTS(f=f)
    >>> tsq.quad(0, 1)
    QuadRes(I=10.000000000000004, err=3.868661047952931e-14, func_calls=73, rec_steps=1)

Note that the function has been called only 73 times.

## Infinite boundary

Infinite boundary condition can be treated efficiently, too.
They are mapped to an integral over a finite interval.
The resulting singularity poses no difficulty for the tanh-sinh method.
The infinite boundary can be specified either as `str` (`'inf'` and `'-inf'`)
or by `math.inf` as well as `numpy.inf`.

As example consider the integrand `1/(1 + (x+1)^2)`.
Its indefinite integral is the `arctan`, so integrating over the whole real
axes yield `pi`.

    >>> import tsquad
    >>> tsq = tsquad.QuadTS(f=f)
    >>> tsq.quad(a='-inf', b='inf')
    QuadRes(I=3.1415926535897643, err=2.3768122480443337e-12, func_calls=528, rec_steps=4)
    >>> import math
    >>> math.pi
    3.141592653589793

## Fourier integral

Integrating oscillatory functions needs special care.
In particular if the bounds are infinite.

Usually, it is possible to integrate over single periods of the oscillation with 
high accuracy.
Summing up the individual integrals yields results for larger intervals at the 
price of summing up the errors, too.
For very rapidly oscillating functions this might cause some trouble.

For infinite bounds this sum of partial integrals becomes infinite, often with
very slow convergence.
The convergence can be accelerated significantly if the terms of the partial 
sum alternate in sign by using the Shanks transform 
(implemented using Wynn's epsilon algorithm).
For Fourier integrals (sin, cos or exp(i ...)) the alternating signs
in the partial sum is realized by summing up finite integrals over **half** the period.

As example consider the half-sided Fourier integral of the algebraically 
decaying function `1/(1+1j*tau)^(s+1)`.
So we aim to integrate

    int_0^inf 1/(1+1j*tau)^(s+1) * exp(1j * w * tau) * d tau .

Note that an analytic expression is possible, which involves the incomplete gamma function
with complex-valued second argument.


    >>> import tsquad
    >>> import mpmath as mp
    >>> s = 0.5
    >>> w = 1
    >>> f = lambda tau, s: 1/(1+1j*tau)**(s+1)
    >>> tsq = tsquad.QuadTS(f=f, args=(s,))
    >>> tsq.quad_Fourier(0, 'inf', w=w)
    QuadRes(I=(1.3040986643460277+0.1523180276515212j), err=None, func_calls=2113, rec_steps=16)
    >>> (-1j*w)**s / 1j**(s+1) * mp.exp(-w) * mp.gammainc(-s, -w-1j*1e-16)
    mpc(real='1.30409866434658424220297858', imag='0.152318027651073881301097481')


# Mathematical Details

Note that the tanh-sinh approach is particularly useful when the integrand is singular at `x=0` 
and the integration goes from `a=0` to `b`, i.e.,

    int_0^b f(x) dx .


To correctly account for a singularity at a different location consider rewriting the 
integrand such that the singularity is located at `x=0`.

The method used here is based on the variable transformation `x -> t` with

    x(t) = b/2(1 - g(t))
    g(t) = tanh(pi/2 * sinh(t))
    ->
    x(t) = b / 2 / (e^(pi/2*sinh(t)) cosh(pi/2 sinh(t)))

which maps the interval `[0, b]` to `[-inf, inf]`,


      int_0^b f(x) dx 
    = int_-inf^inf f(x(t)) b/2 g'(t) dt 
    = int_-inf^inf I(t) dt 
    = dt sum_k f_k w_k
    g'(t) = pi cosh(t)/(2 cosh^2(pi/2 sinh(t))) .

In that way the singularity "occurs" at `t=inf`. 
It has been shown that the error cannot be smaller than `max(|I(t_min)|, |I(t_max)|)`. 
So `t_min` and `t_max` are chosen such that this limitation is below the desired accuracy.
