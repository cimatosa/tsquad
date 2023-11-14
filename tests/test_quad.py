from tsquad import *
import math
import cmath

import logging

logging.root.setLevel(logging.DEBUG)


def test_simple_quad_ts():
    for s in [2, 1, 0.5, -0.5, -0.9, -0.95]:
        f = lambda x: x**s
        a = 0
        b = 3.34

        qts = QuadTS(f=f)
        r = qts._quad(a, b)
        assert abs(r.I - 1 / (s + 1) * b ** (s + 1)) < 1e-12


def test_IntegrationError_tolerance():
    f = lambda x: math.exp(-math.fabs(x)) * math.cos(x)
    a = -1
    b = 30

    try:
        qts = QuadTS(f=f)
        r = qts._quad(a, b)
    except TSIntegrationError:
        pass
    else:
        assert False


def test_recursive_quad():
    f = lambda x: math.cos(x) ** 6 * math.sin(5 * x) ** 4
    a = 0
    b = 300

    F = (
        45 / 512 * math.sin(2 * b)
        + 1 / 64 * math.sin(4 * b)
        + (
            -72072 * math.sin(6 * b)
            - 180180 * math.sin(8 * b)
            - 192192 * math.sin(10 * b)
            - 120120 * math.sin(12 * b)
            - 39468 * math.sin(14 * b)
            + 7
            * (
                429 * math.sin(16 * b)
                + 2860 * math.sin(18 * b)
                + 6
                * (
                    34320 * b
                    + 572 * math.sin(20 * b)
                    + 390 * math.sin(22 * b)
                    + 143 * math.sin(24 * b)
                    + 22 * math.sin(26 * b)
                )
            )
        )
        / 12300288
    )

    qts = QuadTS(f=f)
    r = qts.recursive_quad(a, b)
    assert r.rec_steps > 1
    err = math.fabs(r.I - F)
    assert err < r.err

    try:
        qts = QuadTS(f=f, recursive=False)
        qts.quad_finite_boundary(a, b)
    except TSIntegrationError:
        pass
    else:
        assert False

    qts = QuadTS(f=f, recursive=True)
    r = qts.quad_finite_boundary(a, b)
    assert r.rec_steps > 1
    err = math.fabs(r.I - F)
    assert err < r.err


def test_inf_bound():
    f = lambda x: x**6 * math.exp(-x)

    qts = QuadTS(f=f)
    r = qts.quad_upper_infinite(0)
    err = math.fabs(r.I - 720)
    assert err < r.err

    f = lambda x: x**15 * math.exp(-(x**2))
    qts = QuadTS(f=f)
    r = qts.quad_upper_infinite(0)
    print(r)
    err = math.fabs(r.I - 2520)
    assert err < r.err

    f = lambda x: x**15 * math.exp(-(x**2))
    qts = QuadTS(f=f)
    r = qts.quad_lower_infinite(0)
    print(r)
    err = math.fabs(r.I - -2520)
    assert err < r.err


def test_quad_generic():
    f = lambda x, c, d: c * x**2 * math.exp(-(d * x**2))
    c = 1
    d = 1.0
    qts = QuadTS(f=f, args=(c, d))

    def F(a, b):
        if a == -math.inf:
            if b == math.inf:
                return math.sqrt(math.pi) / 2
            else:
                return (
                    -2 * b * math.exp(-(b**2))
                    + math.sqrt(math.pi) * (1 + math.erf(b))
                ) / 4
        else:
            if b == math.inf:
                return (
                    +2 * a * math.exp(-(a**2))
                    + math.sqrt(math.pi) * (-math.erf(a) + 1)
                ) / 4
            else:
                return (
                    2 * a * math.exp(-(a**2))
                    - 2 * b * math.exp(-(b**2))
                    + math.sqrt(math.pi) * (-math.erf(a) + math.erf(b))
                ) / 4

    for a, b in [
        (-4, -2),
        (0, 3),
        (-math.inf, -1),
        (-1, math.inf),
        (-math.inf, math.inf),
    ]:
        F_ab = F(a, b)
        r = qts.quad(a, b)
        eps = math.fabs(F_ab - r.I)
        assert eps < r.err

        r = qts.quad(b, a)
        eps = math.fabs(-F_ab - r.I)
        assert eps < r.err


def test_quad_osc():
    r_ref_52 = -0.0533008743154537941

    f = lambda x: 1 / (1 + x) * math.sin(3 * x + 2)
    qts = QuadTS(f=f)
    r = qts.quad_osc_finite(0, 52, 2 * math.pi / 3)
    assert abs(r.I - r_ref_52) - r.err

    r_ref_5002 = -0.049406811762728814228
    r = qts.quad_osc_finite(0, 5002, 2 * math.pi / 3)
    assert abs(r.I - r_ref_5002) - r.err

    r = qts.quad_osc_finite(0, 5002, 10 * 2 * math.pi / 3)
    assert abs(r.I - r_ref_5002) < r.err


def test_quad_osc_infinite():
    s = 0.2
    f = lambda x: 1 / x**s * math.cos(x)
    qts = QuadTS(f=f)
    r_ref = math.gamma(1 - s) * math.sin(math.pi * s / 2)
    r = qts.quad_osc_upper_infinite(a=0, period=math.pi)
    assert abs(r.I - r_ref) < 1e-12

    x1 = 5
    x2 = 15
    f_lor = lambda x: 1 / (1 + (x - x1) ** 2) + 1 / (1 + (x - x2) ** 2)
    f = lambda x: f_lor(x) * math.cos(x)

    qts = QuadTS(f=f)
    r_ref = 0.3166409928992998415133346 - 0.8785518615606478863082457
    r = qts.quad_osc_upper_infinite(a=0, period=2 * math.pi)
    assert abs(r.I - r_ref) < 1e-12


def test_quad_osc_generic():
    f = lambda x: math.cos(x) / (1 + (x - 2) ** 2)

    qts = QuadTS(f=f)

    r_ref = -0.51968745377085467319
    r = qts.quad_osc(-1, 5, frequency=1)
    assert abs(r.I - r_ref) < r.err
    r = qts.quad_osc(5, -1, frequency=1)
    assert abs(r.I + r_ref) < r.err

    r_ref = -0.43816779192103931139
    r = qts.quad_osc(-1, math.inf, frequency=1)
    assert abs(r.I - r_ref) < 1e-11
    r = qts.quad_osc(math.inf, -1, frequency=1)
    assert abs(r.I + r_ref) < 1e-11

    r_ref = -0.22918437851152203078
    r = qts.quad_osc(-math.inf, 3, frequency=1)
    assert abs(r.I - r_ref) < 1e-11
    r = qts.quad_osc(3, -math.inf, frequency=1)
    assert abs(r.I + r_ref) < 1e-11

    r_ref = -0.48095228052650475539
    r = qts.quad_osc(-math.inf, math.inf, frequency=1)
    assert abs(r.I - r_ref) < 1e-11
    r = qts.quad_osc(math.inf, -math.inf, frequency=1)
    assert abs(r.I + r_ref) < 1e-11


def test_quad_Fourier():
    s = 0.5
    Gamma_1_5 = 0.8862269254527579
    f = lambda w: w**s * math.exp(-w)
    qts = QuadTS(f=f)
    for t in [0.2, 2.3, 20.4]:
        r = qts.quad_Fourier(0, "inf", -t)
        d = abs(r.I - Gamma_1_5 / (1 + 1j * t) ** (s + 1))
        assert d < 1e-14

        r_c = qts.quad_cos(0, "inf", -t)
        d = abs(r_c.I - (Gamma_1_5 / (1 + 1j * t) ** (s + 1)).real)
        assert d < 1e-14

        r_s = qts.quad_sin(0, "inf", -t)
        d = abs(r_s.I - (Gamma_1_5 / (1 + 1j * t) ** (s + 1)).imag)
        assert d < 1e-14


def test_quad_Fourier_finite():
    for s in [0.5, 1, 2]:
        f = lambda w: w**s

        qts = QuadTS(f=f)
        qts2 = QuadTS(f=f, osc_threshold=0)
        for t in [2, 4, 8, 16, 32]:
            wmax = 5

            r = qts.quad_Fourier(0, wmax, t)
            r2 = qts2.quad_Fourier(0, wmax, t)
            f_Fourier = lambda w: f(w) * cmath.exp(1j * w * t)
            qts_F = QuadTS(f=f_Fourier)
            r_F = qts_F.quad(0, wmax)

            d = abs(r.I - r_F.I)
            assert d < 2e-13

            d2 = abs(r2.I - r_F.I)
            assert d2 < 2e-13
