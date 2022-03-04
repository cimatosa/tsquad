from tsquad import tsquad_py
import cmath
import math


def test_exp():
    """
    integrate exp(z x), over x from 0 to 3.34 with fixed complex z
    """
    z = 1 + 1j
    f = lambda x: cmath.exp(z * x)
    a = 0
    b = 3.34

    qts = tsquad_py.QuadTS(f=f)
    r = qts._quad(a, b)
    r_ref = 1 / z * (cmath.exp(z * b) - cmath.exp(z * a))
    assert abs(r.I - r_ref) < 1e-12


def test_BCF():
    """
    calculate the bath correlation function (BCF) of an (sub-) ohmic spectral density

        bcf = int_0^inf dw J(w) exp(-i w tau)

    for

        J(w) = w^s exp(-w/wc)
    """

    wc = 10
    for s in [0.1, 0.5, 1]:

        J_w = lambda w, s, wc: w ** s * math.exp(-w / wc)
        qts = tsquad_py.QuadTS(f=J_w, args=(s, wc), rec_limit=50)
        bcf = lambda tau, s, wc: (wc / (1 + 1j * wc * tau)) ** (s + 1) * math.gamma(
            s + 1
        )

        for tau in [0.1, 1, 10]:
            r = qts.quad_Fourier(0, math.inf, w=-tau)
            assert abs(r.I - bcf(tau, s, wc)) < 1e-12
