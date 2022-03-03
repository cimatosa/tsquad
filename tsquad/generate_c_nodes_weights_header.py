import ts_parameters as ts
import mpmath as mp

mp.mp.dps = 24


def _s(t, sinh, pi):
    return pi / 2 * sinh(t)


def _1mg(t, exp, cosh, sinh, pi):
    s = _s(t, sinh, pi)
    return 1 / exp(s) / cosh(s)


def _w(t, cosh, sinh, pi):
    return pi / 2 * cosh(t) / cosh(_s(t, sinh, pi)) ** 2


def __write_g_w_c_header(fname):
    print("__write_g_w_c_header is beeing called")
    kwargs = {"min_fixed": 0, "show_zero_exponent": True, "strip_zeros": False}

    N0 = ts._N0
    k = ts._num_sub_grids - 1

    try:
        with open(fname, "w") as f:
            print("#ifndef TS_NODES_WEIGHTS_H\n#define TS_NODES_WEIGHTS_H\n", file=f)
            print("const int N0 = {};".format(ts._N0), file=f)
            print("const int sub_div = {};".format(k), file=f)
            print("const double _tmin = {};\n".format(ts._tmin), end="", file=f)
            print("const int offset[] = {0", end="", file=f)
            os = 2 * N0 + 1
            for ki in range(k):
                print(", {}".format(os), end="", file=f)
                os += N0 * 2 ** (ki + 1)
            print("};", file=f)

            print("const int len[] = {{{}".format(2 * N0 + 1), end="", file=f)
            for ki in range(k):
                print(", {}".format(N0 * 2 ** (ki + 1)), end="", file=f)
            print("};", file=f)
            print(
                "const double _tmax[] = {{{}".format(ts._tmax_list[0]), end="", file=f
            )
            for i in range(1, ts._N_tmax):
                print(", {}".format(ts._tmax_list[i]), end="", file=f)
            print("};\n", end="", file=f)

            print(
                "const unsigned int _N_tmax = {};\n".format(ts._N_tmax), end="", file=f
            )
            print("", file=f)

            s = "static const double dt[{}][{}] = {{{{".format(ts._N_tmax, k + 1)

            for i in range(ts._N_tmax):
                if i == 0:
                    print(s, end="", file=f)
                else:
                    print(",\n{}{{".format(" " * (len(s) - 1)), end="", file=f)
                N = N0
                for ki in range(k + 1):
                    if ki == 0:
                        print(
                            "{:.15e}".format((ts._tmax_list[i] - ts._tmin) / 2 / N),
                            end="",
                            file=f,
                        )
                    else:
                        if ki == 1:
                            comment = "    //t_max {}".format(ts._tmax_list[i])
                        else:
                            comment = ""
                        print(
                            ",{}\n{} {:.15e}".format(
                                comment,
                                " " * (len(s) - 1),
                                (ts._tmax_list[i] - ts._tmin) / 2 / N,
                            ),
                            end="",
                            file=f,
                        )
                    N *= 2
                print("}", end="", file=f)
            print("};\n", file=f)

            s = "static const double one_minus_g[{}][{}] = {{\n".format(
                ts._N_tmax, 2 * N0 * 2 ** k + 1
            )
            s_offs = 2
            for i in range(ts._N_tmax):
                if i == 0:
                    print(s, end="", file=f)
                else:
                    print(",\n", end="", file=f)

                t = mp.linspace(ts._tmin, ts._tmax_list[i], 2 * N0 * 2 ** k + 1)
                N = N0
                for ki in range(k + 1):
                    if ki == 0:
                        print("{" + " " * s_offs, end="", file=f)
                        newline = ""
                        comment = "    // t_max = {}".format(ts._tmax_list[i])
                        step = 2 ** (k - ki)
                        t_slice = t[::step]
                        assert len(t_slice) == 2 * N + 1
                        assert t_slice[0] == t[0]
                        assert t_slice[-1] == t[-1]
                    else:
                        print(
                            ",\n" + " " * s_offs + "  // {}. layer".format(ki),
                            end="",
                            file=f,
                        )
                        step = 2 ** (k - ki + 1)
                        t_slice = t[step // 2 :: step]
                        assert len(t_slice) == N
                    N *= 2

                    for i, ti in enumerate(t_slice):
                        print(
                            newline,
                            mp.nstr(
                                _1mg(ti, mp.exp, mp.cosh, mp.sinh, mp.pi), 17, **kwargs
                            ),
                            end="",
                            file=f,
                        )
                        newline = ",{}\n".format(comment) + " " * (s_offs + 1)
                        comment = ""
                print("}", end=(""), file=f)
            print("};\n", file=f)

            s = "static const double w[{}][{}] = {{".format(
                ts._N_tmax, 2 * N0 * 2 ** k + 1
            )
            s_offs = 2

            for i in range(ts._N_tmax):
                if i == 0:
                    print(s, end="", file=f)
                else:
                    print(",\n", end="", file=f)

                t = mp.linspace(ts._tmin, ts._tmax_list[i], 2 * N0 * 2 ** k + 1)
                N = N0
                for ki in range(k + 1):
                    if ki == 0:
                        print("{" + " " * s_offs, end="", file=f)
                        newline = ""
                        comment = "    // t_max = {}".format(ts._tmax_list[i])
                        step = 2 ** (k - ki)
                        t_slice = t[::step]
                        assert len(t_slice) == 2 * N + 1
                        assert t_slice[0] == t[0]
                        assert t_slice[-1] == t[-1]
                    else:
                        print(
                            ",\n" + " " * s_offs + "  // {}. layer".format(ki),
                            end="",
                            file=f,
                        )
                        step = 2 ** (k - ki + 1)
                        t_slice = t[step // 2 :: step]
                        assert len(t_slice) == N
                    N *= 2

                    for i, ti in enumerate(t_slice):
                        print(
                            newline,
                            mp.nstr(_w(ti, mp.cosh, mp.sinh, mp.pi), 17, **kwargs),
                            end="",
                            file=f,
                        )
                        newline = ",{}\n".format(comment) + " " * (s_offs + 1)
                        comment = ""
                print("}", end=(""), file=f)
            print("};\n", file=f)

            s = "static const double y_over_1_minus_y[{}][{}] = {{\n".format(
                ts._N_tmax, 2 * N0 * 2 ** k + 1
            )
            s_offs = 2
            for i in range(ts._N_tmax):
                if i == 0:
                    print(s, end="", file=f)
                else:
                    print(",\n", end="", file=f)

                t = mp.linspace(ts._tmin, ts._tmax_list[i], 2 * N0 * 2 ** k + 1)
                N = N0
                for ki in range(k + 1):
                    if ki == 0:
                        print("{" + " " * s_offs, end="", file=f)
                        newline = ""
                        comment = "    // t_max = {}".format(ts._tmax_list[i])
                        step = 2 ** (k - ki)
                        t_slice = t[::step]
                        assert len(t_slice) == 2 * N + 1
                        assert t_slice[0] == t[0]
                        assert t_slice[-1] == t[-1]
                    else:
                        print(
                            ",\n" + " " * s_offs + "  // {}. layer".format(ki),
                            end="",
                            file=f,
                        )
                        step = 2 ** (k - ki + 1)
                        t_slice = t[step // 2 :: step]
                        assert len(t_slice) == N
                    N *= 2

                    for i, ti in enumerate(t_slice):
                        x = _1mg(ti, mp.exp, mp.cosh, mp.sinh, mp.pi)
                        print(
                            newline, mp.nstr(x / (1 - x), 17, **kwargs), end="", file=f
                        )
                        newline = ",{}\n".format(comment) + " " * (s_offs + 1)
                        comment = ""
                print("}", end=(""), file=f)
            print("};\n", file=f)

            print("#endif", file=f)
    except Exception as e:
        print("FAILED to successfully write {}".format(fname))
        import os

        os.remove(fname)
        raise


if __name__ == "__main__":
    __write_g_w_c_header(fname="ts_nodes_weights.h")
