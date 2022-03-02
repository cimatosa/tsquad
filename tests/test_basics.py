import numpy as np
import math
from tsquad import tsquad_py


def test_pass_class_instance_by_reference():
    """
    ensure that the result object is passed as reference, such it can be modified within another function
    """

    def inc(r):
        r.func_calls = 1

    r = tsquad_py.QuadRes(I=0, err=0, func_calls=0, rec_steps=0)

    assert r.func_calls == 0
    inc(r)
    assert r.func_calls == 1

def test_add_QuadRes():
    r1 = tsquad_py.QuadRes(I=1, err=2, func_calls=3, rec_steps=4)
    r2 = r1 + r1

    assert r1.I == 1
    assert r1.err == 2
    assert r1.func_calls == 3
    assert r1.rec_steps == 4

    assert r2.I == 2
    assert r2.err == 4
    assert r2.func_calls == 6
    assert r2.rec_steps == 8

def test_inf():
    a = np.inf
    assert math.isinf(a)

    b = math.inf
    assert math.isinf(b)

    c = -np.inf
    assert math.isinf(c)

    assert np.inf == math.inf
    assert -np.inf == -math.inf
    assert np.inf != -math.inf
    assert -np.inf != math.inf