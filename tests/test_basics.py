from tsquad import tsquad_py

def test_pass_class_instance_by_reference():
    """
    ensure that the result object is passed as reference, such it can be modified within another function
    """

    def inc(r):
        r.func_calls = 1

    r = tsquad_py.QuadRes(I=0, err=0, func_calls=0, adaptive_splits=0)

    assert r.func_calls == 0
    inc(r)
    assert r.func_calls == 1