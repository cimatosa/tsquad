import mpmath as mp

from .ts_parameters import t_max_list, t_min, N_t_max, N_0, num_sub_grids
mp.mp.dps = 24

def _s(t, sinh, pi):
    return pi/2 * sinh(t)

def _1mg(t, exp, cosh, sinh, pi):
    s = _s(t, sinh, pi)
    return 1/exp(s)/cosh(s)

def _w(t, cosh, sinh, pi):
    return pi/2 * cosh(t) / cosh(_s(t, sinh, pi))**2

def __show_g_w_tmax(_tmax_list):
    for tm in _tmax_list:
        print("tmax {}".format(tm))
        print("  g: ", mp.nstr(_1mg(tm, mp.exp, mp.cosh, mp.sinh, mp.pi), 17))
        print("  w: ", mp.nstr(_w(tm, mp.cosh, mp.sinh, mp.pi), 17))

def __write_g_w_(N0, k, fname, _tmax_list, _tmin):
    kwargs = {'min_fixed' : 0,
              'show_zero_exponent' : True,
              'strip_zeros': False}

    _N_tmax = len(_tmax_list)

    with open(fname, 'w') as f:
        for i in range(_N_tmax):
            if i == 0:
                print("dt = [[", end='', file=f)
            else:
                print(",\n      [", end='', file=f)
            N = N0
            for ki in range(k):
                print("{:.15e}, ".format((_tmax_list[i] - _tmin) / 2 / N), end='', file=f)
                N *= 2
            print("{:.15e}]".format((_tmax_list[i] - _tmin) / 2 / N), end='', file=f)
        print("]", file=f)


        for i in range(_N_tmax):
            if i == 0:
                print("_1mg = [", end='', file=f)
            else:
                print(",\n        ", end='', file=f)

            t = mp.linspace(_tmin, _tmax_list[i], 2*N0*2**k+1)
            N = N0
            for ki in range(k+1):
                if ki == 0:
                    print("[", end='', file=f)
                    step = 2 ** (k - ki)
                    t_slice = t[::step]
                    assert len(t_slice) == 2*N+1
                    assert t_slice[0]  == t[0]
                    assert t_slice[-1] == t[-1]
                else:
                    print(",\n         ", end='', file=f)
                    step = 2 ** (k - ki + 1)
                    t_slice = t[step//2::step]
                    assert len(t_slice) == N
                N *= 2


                for i, ti in enumerate(t_slice):
                    if i == 0:
                        print("[", mp.nstr(_1mg(ti, mp.exp, mp.cosh, mp.sinh, mp.pi), 17, **kwargs), end='', file=f)
                    else:
                        print(",\n          ", mp.nstr(_1mg(ti, mp.exp, mp.cosh, mp.sinh, mp.pi), 17, **kwargs), end='', file=f)
                print("]", end=(''), file=f)
            print("]", end=(''), file=f)
        print("]", file=f)

        for i in range(_N_tmax):
            if i == 0:
                print("_w = [", end='', file=f)
            else:
                print(",\n      ", end='', file=f)

            t = mp.linspace(_tmin, _tmax_list[i], 2 * N0 * 2 ** k + 1)
            N = N0
            for ki in range(k + 1):
                if ki == 0:
                    print("[", end='', file=f)
                    step = 2 ** (k - ki)
                    t_slice = t[::step]
                    assert len(t_slice) == 2 * N + 1
                    assert t_slice[0] == t[0]
                    assert t_slice[-1] == t[-1]
                else:
                    print(",\n       ", end='', file=f)
                    step = 2 ** (k - ki + 1)
                    t_slice = t[step // 2::step]
                    assert len(t_slice) == N
                N *= 2

                for i, ti in enumerate(t_slice):
                    if i == 0:
                        print("[", mp.nstr(_w(ti, mp.cosh, mp.sinh, mp.pi), 17, **kwargs), end='', file=f)
                    else:
                        print(",\n        ", mp.nstr(_w(ti, mp.cosh, mp.sinh, mp.pi), 17, **kwargs), end='',
                              file=f)
                print("]", end=(''), file=f)
            print("]", end=(''), file=f)
        print("]", file=f)

if __name__ == "__main__":
    __show_g_w_tmax(_tmax_list)
    #__write_g_w_(N0=4, k=_num_sub_grids - 1, fname=_fname_abs)