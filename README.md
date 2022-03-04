# tsquad - Numeric Integration Using tanh-sinh Variable Transformation

The `tsquad` package provides general purpose integration routines.
For simple integration tasks this package is as good as the standard routines
available e.g. by `scipy`.
However, it is particularly suited to efficiently handle **singularities** by exploiting
the *tanh-sinh* variable transformation (also known as *double exponential* approach), as nicely described here http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/dhb-tanh-sinh.pdf.
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

Add `tsquad` to your `pyproject.toml` file, like that
    
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









# Mathematical Details

Note that this approach is particularly useful when the integrand is singular at `x=0` 
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
