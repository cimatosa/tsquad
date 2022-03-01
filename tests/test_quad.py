from tsquad import tsquad_py

def test_simple_quad_ts():
    for s in [2, 1, 0.5, -0.5, -0.9, -0.95]:
        f = lambda x: x**s
        a = 0
        b = 3.34

        r = tsquad_py._quad_ts(f=f, a=a, b=b, args=tuple(), abs_tol=1e-12, rel_tol=1e-12, force_t_max_idx=None, subgrid_max=6)
        assert abs(r.I - 1/(s+1)*b**(s+1)) < 1e-12