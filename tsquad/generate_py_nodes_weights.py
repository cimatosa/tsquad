"""
generate the nodes and weights for the Tanh-Sinh integration scheme

Use high precision mpmath library to pre-calculate the nodes and weight
and save them to a file.

the principal variable transformation x -> t which maps x in [0, b] to t in [-inf, inf]
    x(t) = b/2(1 - g(t))
    g(t) = tanh(pi/2 * sinh(t))

See `ts_parameters.py` for parameters to control the pre-calculation
"""

# python imports
import logging
import os

# third party imports
import mpmath as mp

# tsquad module imports
from . import tsconfig

mp.mp.dps = 24


def _s(t):
    """helper function proportional to sinh"""
    return mp.pi / 2 * mp.sinh(t)


def _1mg(t):
    """the function 1-g(t) rewritten to yield accurate results for t -> inf, i.e., 0 < 1-g(t) << 1"""
    s = _s(t)
    return 1 / mp.exp(s) / mp.cosh(s)


def _w(t):
    """the weight which originates from g'(t)"""
    return mp.pi / 2 * mp.cosh(t) / mp.cosh(_s(t)) ** 2


def print_1mg_w_t_max(t_max_list):
    """
    Print the argument 1-g(t) and weight w(t) for t at the right boundary of the truncated infinite integral.
    Use the various values for that boundary given by the list `t_max_list`.
    """
    for tm in t_max_list:
        print("t_max {}".format(tm))
        print("  1-g(t_max): ", mp.nstr(_1mg(tm), 17))
        print("    w(t_max): ", mp.nstr(_w(tm), 17))


def write_g_w_(N_0, k, f_name, t_max_list, t_min):
    kwargs = {"min_fixed": 0, "show_zero_exponent": True, "strip_zeros": False}

    N_t_max = len(t_max_list)

    with open(f_name, "w") as f:
        for i in range(N_t_max):
            if i == 0:
                print("dt = [[", end="", file=f)
            else:
                print(",\n      [", end="", file=f)
            N = N_0
            for ki in range(k):
                print(
                    "{:.15e}, ".format((t_max_list[i] - t_min) / 2 / N), end="", file=f
                )
                N *= 2
            print("{:.15e}]".format((t_max_list[i] - t_min) / 2 / N), end="", file=f)
        print("]", file=f)

        for i in range(N_t_max):
            if i == 0:
                print("_1mg = [", end="", file=f)
            else:
                print(",\n        ", end="", file=f)

            t = mp.linspace(t_min, t_max_list[i], 2 * N_0 * 2 ** k + 1)
            N = N_0
            for ki in range(k + 1):
                if ki == 0:
                    print("[", end="", file=f)
                    step = 2 ** (k - ki)
                    t_slice = t[::step]
                    assert len(t_slice) == 2 * N + 1
                    assert t_slice[0] == t[0]
                    assert t_slice[-1] == t[-1]
                else:
                    print(",\n         ", end="", file=f)
                    step = 2 ** (k - ki + 1)
                    t_slice = t[step // 2 :: step]
                    assert len(t_slice) == N
                N *= 2

                for i, ti in enumerate(t_slice):
                    if i == 0:
                        print("[", mp.nstr(_1mg(ti), 17, **kwargs), end="", file=f)
                    else:
                        print(
                            ",\n          ",
                            mp.nstr(_1mg(ti), 17, **kwargs),
                            end="",
                            file=f,
                        )
                print("]", end=(""), file=f)
            print("]", end=(""), file=f)
        print("]", file=f)

        for i in range(N_t_max):
            if i == 0:
                print("_w = [", end="", file=f)
            else:
                print(",\n      ", end="", file=f)

            t = mp.linspace(t_min, t_max_list[i], 2 * N_0 * 2 ** k + 1)
            N = N_0
            for ki in range(k + 1):
                if ki == 0:
                    print("[", end="", file=f)
                    step = 2 ** (k - ki)
                    t_slice = t[::step]
                    assert len(t_slice) == 2 * N + 1
                    assert t_slice[0] == t[0]
                    assert t_slice[-1] == t[-1]
                else:
                    print(",\n       ", end="", file=f)
                    step = 2 ** (k - ki + 1)
                    t_slice = t[step // 2 :: step]
                    assert len(t_slice) == N
                N *= 2

                for i, ti in enumerate(t_slice):
                    if i == 0:
                        print("[", mp.nstr(_w(ti), 17, **kwargs), end="", file=f)
                    else:
                        print(
                            ",\n        ", mp.nstr(_w(ti), 17, **kwargs), end="", file=f
                        )
                print("]", end=(""), file=f)
            print("]", end=(""), file=f)
        print("]", file=f)


########################################################################################################################
##    pre-calculate nodes and weights if needed
########################################################################################################################


def run(overwrite=False):
    pth, fl = os.path.split(__file__)
    _f_name_abs = os.path.join(pth, tsconfig._f_name)

    # generate, if missing, using mpmath routines
    if not os.path.exists(_f_name_abs):
        logging.info("pre-calculated nodes/weights missing ({})".format(_f_name_abs))
    elif overwrite:
        logging.info("overwrite existing file {}".format(_f_name_abs))
    else:
        return

    logging.info("generate nodes and weights ...")
    write_g_w_(
        N_0=tsconfig.N_0,
        k=tsconfig.num_sub_grids - 1,
        f_name=_f_name_abs,
        t_max_list=tsconfig.t_max_list,
        t_min=tsconfig.t_min,
    )
    logging.info("done!")


if __name__ == "__main__":
    print_1mg_w_t_max(tsconfig.t_max_list)
    run(overwrite=True)
