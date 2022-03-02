from tsquad import tsquad_py
import mpmath
import math


def test_simple_quad_ts():
    for s in [2, 1, 0.5, -0.5, -0.9, -0.95]:
        f = lambda x: x ** s
        a = 0
        b = 3.34

        tsq = tsquad_py.QuadTS(f=f)
        r = tsq._quad(a, b)
        assert abs(r.I - 1 / (s + 1) * b ** (s + 1)) < 1e-12


def test_IntegrationError_tolerance():
    f = lambda x: math.exp(-math.fabs(x)) * math.cos(x)
    a = -1
    b = 30

    try:
        tsq = tsquad_py.QuadTS(f=f)
        r = tsq._quad(a, b)
    except tsquad_py.TSIntegrationError:
        pass
    else:
        assert False


def test_recursive_quad():
    f = lambda x: math.cos(x)**6 * math.sin(5*x)**4
    a = 0
    b = 300

    F = 45 / 512 * math.sin(2 * b) + 1 / 64 * math.sin(4 * b) + \
        (-72072 * math.sin(6 * b) - 180180 * math.sin(8 * b) - 192192 * math.sin(10 * b) - 120120 * math.sin(12 * b)
         - 39468 * math.sin(14 * b) + 7 * (429 * math.sin(16 * b) + 2860 * math.sin(18 * b) + 6 * (34320 * b
         + 572 * math.sin(20 * b) + 390 * math.sin(22 * b) + 143 * math.sin(24 * b) + 22 * math.sin(26 * b)))) / 12300288

    tsq = tsquad_py.QuadTS(f=f)
    r = tsq.recursive_quad(a, b)
    assert r.rec_steps > 1
    err = math.fabs(r.I - F)
    assert err < r.err

    try:
        tsq = tsquad_py.QuadTS(f=f, recursive=False)
        tsq.quad_finite_boundary(a, b)
    except tsquad_py.TSIntegrationError:
        pass
    else:
        assert False

    tsq = tsquad_py.QuadTS(f=f, recursive=True)
    r = tsq.quad_finite_boundary(a, b)
    assert r.rec_steps > 1
    err = math.fabs(r.I - F)
    assert err < r.err


def test_inf_bound():
        f = lambda x: x**6 * math.exp(-x)

        tsq = tsquad_py.QuadTS(f=f)
        r = tsq.quad_upper_infinite(0)
        err = math.fabs(r.I - 720)
        assert err < r.err

        f = lambda x: x ** 15 * math.exp(-x**2)
        tsq = tsquad_py.QuadTS(f=f)
        r = tsq.quad_upper_infinite(0)
        print(r)
        err = math.fabs(r.I - 2520)
        assert err < r.err

        f = lambda x: x ** 15 * math.exp(-x**2)
        tsq = tsquad_py.QuadTS(f=f)
        r = tsq.quad_lower_infinite(0)
        print(r)
        err = math.fabs(r.I - -2520)
        assert err < r.err

def test_quad_generic():
    f = lambda x: x ** 2 * math.exp(-x**2)
    tsq = tsquad_py.QuadTS(f=f)

    def F(a, b):
        if a == -math.inf:
            if b == math.inf:
                return math.sqrt(math.pi)/2
            else:
                return (- 2 * b * math.exp(-b ** 2) + math.sqrt(math.pi) * (1 + math.erf(b)))/4
        else:
            if b == math.inf:
                return (+ 2 * a * math.exp(-a ** 2) + math.sqrt(math.pi) * (-math.erf(a) + 1)) / 4
            else:
                return (2*a*math.exp(-a**2) - 2*b*math.exp(-b**2) + math.sqrt(math.pi)*(-math.erf(a) + math.erf(b)))/4

    for a,b in [(-4, -2), (0, 3), (-math.inf, -1), (-1, math.inf), (-math.inf, math.inf)]:
        F_ab = F(a, b)
        r = tsq.quad(a, b)
        eps = math.fabs(F_ab - r.I)
        assert eps < r.err

        r = tsq.quad(b, a)
        eps = math.fabs(-F_ab - r.I)
        assert eps < r.err
