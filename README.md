# tsquad - Numeric Integration Using Tanh-Sinh Variable Transformation

This implementation is based on notes from 
http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/dhb-tanh-sinh.pdf.

# Install

# Examples

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
