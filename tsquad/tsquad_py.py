"""
    Numeric Integration Using Tanh-Sinh Variable Transformation

    This is the main module which implements the Tanh-Sinh method for numeric integration.
"""

# python import
import logging
import os
import sys
import traceback
import typing

# third party import
import numpy as np

# tsquad module imports
from . import tsconfig
from . import generate_py_nodes_weights
generate_py_nodes_weights.run()
from . import nodes_weights

########################################################################################################################
##    typedefs
########################################################################################################################
numeric = typing.Union[int, float]


########################################################################################################################
##    define module specific exceptions
########################################################################################################################


class TSIntegrationError(Exception):
    pass


class TSIntegrationLimitReachedError(Exception):
    pass


class TSIntegrationFunctionEvaluationError(Exception):
    pass


class TSOscIntegrationLimitReachedError(Exception):
    pass


########################################################################################################################
##    implementation of the tanh-sinh method
########################################################################################################################


def _f_x_exception_wrapper(f, x, args):
    try:
        f_x = f(x, *args)
    except Exception as e:
        logging.error(
            "calling function at x={:.8e} failed with exception {}, show traceback:\n".format(
                x, e.__class__.__name__
            )
        )
        traceback.print_exc()
        raise TSIntegrationFunctionEvaluationError(
            "Failed to evaluate function (Exception occurred during function call)"
        )

    if np.isnan(f_x):
        raise TSIntegrationFunctionEvaluationError(
            "Failed to evaluate function (function returns nan at x={:.8e})".format(x)
        )
    return f_x


def _get_integral_bound(f, a, b, param, tol):
    """
        chose t_max such that |w_(t_max) I(g(t_max))| < tol
    """
    sc = (b - a) / 2
    for i in range(tsconfig.N_t_max):
        f_x = _f_x_exception_wrapper(f, a + sc * nodes_weights._1mg[i][0][-1], param)
        tmp = abs(sc * f_x * nodes_weights._w[i][0][-1])
        if tmp < tol:
            return i


class QuadRes(object):
    __slots__ = ('I', 'err', 'func_calls', 'adaptive_splits')

    def __init__(self, I=0, err=0, func_calls=0, adaptive_splits=0):
        self.I = I
        self.err = err
        self.func_calls = func_calls
        self.adaptive_splits = adaptive_splits

    def __str__(self):
        return ('QuadRes(I={}, err={}, func_calls={}, adaptive_splits={}'.format(self.I, self.err, self.func_calls, self.adaptive_splits))
    


def quad_ts(
    f,
    a,
    b,
    args=tuple(),
    abs_tol=1e-12,
    rel_tol=1e-12,
    rec_limit=50,
    force_t_max_idx=None,
    adaptive=True,
    subgrid_max=0,
    debug=False,
):
    """
    Integrate f(x, *args) from a to b. If a = 0 singularities at x = 0 are treated very well.
    The procedure is assumed to have converged if the estimated error `err_est` and the
    value of the integral `I` fulfill:

        err_est < abs_tol or err_est / |I| < rel_tol

    Note that the routine starts with a t-grid using `2 * tsconfig.N_0 + 1` equally spaced nodes on the t-axes.
    If the tolerance criterion is not met, use a finer grind by doubling N_0.
    Repeat this at most `subgrid_max` times. Be aware that this requires that the nodes and weights need to be
    pre-calculated for the finer grids, i.e., `subgrid_max <= tsconfig.num_sub_grids` needs to hold.
    If you change the values in `tsconfig.py` simply run `python3 generate_py_nodes_weights` to force an
    update of the `nodes_weights.py` file.

    If the tolerance criterion os not met, repeat recursively on sub-intervals.
    The maximum number of recursions is given by `rec_limit`.

    Note that the integral over [a, 0] is automatically translated to -I([0, a])
    in order to better treat a potential singularity at x=0.

    :param f: function to integrate, callable of the form f(x, *args)
    :param a: lower integral boundary
    :param b: upper integral boundary
    :param args: arguments passed to `f`
    :param abs_tol: absolute tolerance
    :param rel_tol: relative tolerance
    :param rec_limit: maximum number of recursions allowed
    :param force_t_max_idx: Set the upper boundary/truncation for the (transformed) t-integral by hand,
                            this controls the resolution of the lower bound a of the x-integral.
                            Mainly needed by testing purposes.
    :param adaptive: if True, allow for recursive application of the tanh-sinh scheme
    :param subgrid_max: Set the number of sub-grids to use. If `subgrid_max=0` use the largest number ava
    :param debug:
    :return: a QuadRes result object where `I` contains the value of the numeric integration and `err` the estimate of
    the error. In addition, `func_calls` the `adaptive_splits` is provided by the result object. 
    """

    if subgrid_max == 0:
        subgrid_max = tsconfig.num_sub_grids - 1
    elif subgrid_max < 3:
        print("subgrid_max ({}) set to 3".format(subgrid_max))
        subgrid_max = 3
    elif subgrid_max > tsconfig.num_sub_grids - 1:
        print(
            "subgrid_max ({}) must not exceed {}".format(
                subgrid_max, tsconfig.num_sub_grids - 1
            )
        )
        raise TSIntegrationError(
            "ts_quad integration error: subgrid_max exceeds pre-calculated nodes and weights"
        )

    func_calls = [0]
    kwargs = {
        "func_calls": func_calls,
        "force_t_max_idx": force_t_max_idx,
        "subgrid_max": subgrid_max,
        "debug": debug,
    }

    if adaptive:
        quad_method = _adaptive_quad_ts
        kwargs["rec_limit"] = rec_limit
    else:
        quad_method = _quad_ts

    if a == b:
        return QuadRes(I=0, err=0, func_calls=0, adaptive_splits=0)

    elif a == -np.inf:
        if b == np.inf:
            # both inf, correct order
            res_1 = _quad_ts_minus_inf_b(
                quad_method, f, 0, args, abs_tol, rel_tol, **kwargs
            )
            res_2 = _quad_ts_a_plus_inf(
                quad_method, f, 0, args, abs_tol, rel_tol, **kwargs
            )
            res = res_1 + res_2
        else:
            # a=-inf, b is finite
            res = _quad_ts_minus_inf_b(
                quad_method, f, b, args, abs_tol, rel_tol, **kwargs
            )
    elif a == np.inf:
        if b == -np.inf:
            # both inf, wrong order
            res_1 = _quad_ts_minus_inf_b(
                quad_method, f, 0, args, abs_tol, rel_tol, **kwargs
            )
            res_2 = _quad_ts_a_plus_inf(
                quad_method, f, 0, args, abs_tol, rel_tol, **kwargs
            )
            res = res_1 + res_2
            res[0] *= -1
        else:
            # a=inf, b is finite
            res = _quad_ts_a_plus_inf(
                quad_method, f, b, args, abs_tol, rel_tol, **kwargs
            )
            res[0] *= -1
    elif b == np.inf:
        # a finite, b +inf, correct order
        res = _quad_ts_a_plus_inf(quad_method, f, a, args, abs_tol, rel_tol, **kwargs)
    elif b == -np.inf:
        # a finite, b - inf, wrong order, do I([-inf, a]) and return its negative
        res = _quad_ts_minus_inf_b(quad_method, f, a, args, abs_tol, rel_tol, **kwargs)
        res[0] *= -1
    elif b == 0:
        # map I([a, 0]) to -I([0, a]), even for a < 0
        res = quad_method(f, 0, a, args, abs_tol, rel_tol, **kwargs)
        res[0] *= -1
    else:
        res = quad_method(f, a, b, args, abs_tol, rel_tol, **kwargs)

    if adaptive:
        adaptive_splits = res[2]
        res = res[0:2]
    else:
        adaptive_splits = 0

    if get_func_calls:
        res = np.asarray([res[0], res[1], func_calls[0]], dtype=object)

    if get_adaptive_splits:
        res = np.asarray([_r for _r in res] + [adaptive_splits], dtype=object)

    return res


def _quad_ts_a_plus_inf(quad_method, f, a, args, abs_tol, rel_tol, **kwargs):
    res1 = quad_method(
        lambda t: f(a + t, *args), 0, 1, tuple(), abs_tol / 2, rel_tol / 2, **kwargs
    )
    res2 = quad_method(
        lambda t: f(a + 1 / t, *args) / t ** 2,
        0,
        1,
        tuple(),
        abs_tol / 2,
        rel_tol / 2,
        **kwargs
    )
    return res1 + res2


def _quad_ts_minus_inf_b(quad_method, f, b, args, abs_tol, rel_tol, **kwargs):
    res1 = quad_method(
        lambda t: f(b - t, *args), 0, 1, tuple(), abs_tol / 2, rel_tol / 2, **kwargs
    )
    res2 = quad_method(
        lambda t: f(b - 1 / t, *args) / t ** 2,
        0,
        1,
        tuple(),
        abs_tol / 2,
        rel_tol / 2,
        **kwargs
    )
    return res1 + res2


def _quad_ts(
    f: typing.Callable,
    a: numeric,
    b: numeric,
    args: tuple,
    abs_tol: float,
    rel_tol: float,
    force_t_max_idx: [None, int],
    subgrid_max: int,
    debug: bool = False,
) -> QuadRes:
    """
        helper function that performs the actual Tanh-Sinh integration
    """

    func_calls = 0
    if force_t_max_idx is None:
        t_max_idx = _get_integral_bound(f, a, b, args, abs_tol)
        func_calls += t_max_idx + 1
    else:
        t_max_idx = force_t_max_idx

    data_dt = nodes_weights.dt[t_max_idx]
    data_1mg = nodes_weights._1mg[t_max_idx]
    data_w = nodes_weights._w[t_max_idx]

    eps = 10 ** -14

    I_res_n2 = 0
    I_res_n1 = 0
    I_res = 0
    sum_tmp = 0

    sc = (b - a) / 2

    f_x_max = _f_x_exception_wrapper(f, a + sc * data_1mg[0][+0], args)
    w_f_t_max = sc * f_x_max * data_w[0][0]
    f_x_min = _f_x_exception_wrapper(f, a + sc * data_1mg[0][-1], args)
    w_f_t_min = sc * f_x_min * data_w[0][-1]

    func_calls += 2

    d4_t_min = abs(w_f_t_min)
    d4_t_max = abs(w_f_t_max)
    err4 = max(d4_t_min, d4_t_max)

    err_est = np.nan
    err1 = err2 = err3 = np.nan
    N = subgrid_max
    assert N >= 2

    if debug:
        print("##  TS integration in debug mode ##")
        print("## " + "-" * 29 + " ##")
        print("tmin", a + sc * data_1mg[0][-1])
        print("tmax", a + sc * data_1mg[0][+0])
        print("f(t_min)", f_x_min)
        print("f(t_max)", f_x_max)
        print("d4_tmin", d4_t_min)
        print("d4_tmax", d4_t_max)
        print("## " + "-" * 29 + " ##")

    success = False
    may_be_success = False
    I_res_final = 0
    err_est_final = 0

    for n in range(N + 1):
        max_tmp = 0
        for k in range(len(data_w[n])):
            f_x = _f_x_exception_wrapper(f, a + sc * data_1mg[n][k], args)
            w_f = sc * f_x * data_w[n][k]

            max_tmp = max(max_tmp, abs(w_f))
            sum_tmp += w_f
        func_calls += len(data_w[n])
        I_res_n2 = I_res_n1
        I_res_n1 = I_res
        I_res = sum_tmp * data_dt[n]
        err3 = abs(eps * max_tmp)

        if debug:
            print("n", n, "I_n", I_res)

        if n >= 2:
            d1 = abs(I_res - I_res_n1)
            if debug:
                print("d1 = I_n - I_(n-1)    {:.8e} -> err {:.16e}".format(d1, d1 ** 2))
            if d1 == 0:
                if debug:
                    print("best we can have!")
                    print("return, {:.16e} +- {:.4e}".format(I_res, err_est))
                err_est = max(err3, err4)
                return QuadRes(I=I_res, err=err_est, func_calls=func_calls, adaptive_splits=0)
            else:
                d1_log = np.log10(d1)

            d2 = abs(I_res - I_res_n2)
            if debug:
                print("d2 = I_n - I_(n-2)    {:.8e}".format(d2))

            err1 = d1 ** 2
            if debug:
                print("err1 = {:.8e}".format(err1))

            if d2 > 1e-308:
                d2_log = np.log10(d2)
                tmp = d1_log ** 2 / d2_log
                if debug:
                    print("d1_log", d1_log)
                    print("d2_log", d2_log)
                    print("tmp   ", tmp)
                if tmp < -308:
                    err2 = 0
                    if debug:
                        print("err2 = 0  (due to 10 ** d1log^2/d2log underflow")
                elif tmp > 308:
                    err2 = 10
                    if debug:
                        print("err2 = 10  (due to 10 ** d1log^2/d2log overflow")
                else:
                    err2 = 10 ** tmp
                    if debug:
                        print("err2 = {:.8e}".format(err2))

            else:
                err2 = 10
                if debug:
                    print("err2 = 10  (due to d2 < 1e-308)")

            if debug:
                print("err3 = {:.8e}".format(err3))

            if err2 >= 10:
                if debug:
                    print("err1 >= 10  ->  continue")
                may_be_success = False
                continue

            err_est = max(err1, err2, err3, err4)

            if (err_est < abs_tol) or (err_est < rel_tol * abs(I_res)):
                # the error constrain is fulfilled
                if may_be_success:
                    # the error constrain has been fulfilled twice in a row, so we can quit here

                    # we set success and save the current results only if we have not already succeeded
                    # (might be the case in debug mode to see convergence properties)
                    if not success:
                        success = True
                        I_res_final = I_res
                        err_est_final = err_est
                    # is not in debug mode exit loop here
                    if not debug:
                        break
                    else:
                        print(
                            "err constrain fulfilled second time in a row, BUT keep in looping due to debug mode"
                        )
                else:
                    # first time we fulfilled error constrain
                    may_be_success = True
                    if debug:
                        print(
                            "err constrains fulfilled first time (may_be_success set to True)"
                        )

            else:
                # when ever the error constrain is not fulfilled
                # we reset may_be_success
                may_be_success = False
                if debug:
                    print("err constrains NOT fulfilled (may_be_success set to False)")

    if success:
        if debug:
            print("return, {:.16e} +- {:.4e}".format(I_res_final, err_est_final))
        return QuadRes(I=I_res_final, err=err_est_final, func_calls=func_calls, adaptive_splits=0)

    raise TSIntegrationError(
        "Required tolerance not achieved!\n"
        + "err_est       = {} > abs_tol = {}\n".format(err_est, abs_tol)
        + "err_est/I_res = {} > rel_tol = {}\n".format(err_est / I_res, abs_tol)
        + "Reached max number N={} of sub grids\n".format(N)
        + "tmax: {}\n".format(tsconfig.t_max_list[t_max_idx])
        + "err: d1^2/d2: {}\n".format(err2)
        + "err:     2d1: {}\n".format(err1)
        + "err:      d3: {}\n".format(err3)
        + "err:      d4: {}".format(err4)
    )


def _split_quad_ts(
    f, a, b, args, abs_tol, rel_tol, cnt, func_calls, force_tmax_idx, subgrid_max, debug
):
    res1 = _step_quad_ts(
        f,
        a,
        (a + b) / 2,
        args,
        abs_tol,
        rel_tol,
        cnt,
        func_calls,
        force_tmax_idx,
        subgrid_max,
        debug,
    )
    res2 = _step_quad_ts(
        f,
        (a + b) / 2,
        b,
        args,
        abs_tol,
        rel_tol,
        cnt,
        func_calls,
        force_tmax_idx,
        subgrid_max,
        debug,
    )
    return res1 + res2


def _step_quad_ts(
    f, a, b, args, abs_tol, rel_tol, cnt, func_calls, force_tmax_idx, subgrid_max, debug
):
    cnt[0] = cnt[0] - 1
    if cnt[0] == 0:
        raise TSIntegrationLimitReachedError()
    try:
        tmp = _quad_ts(
            f,
            a,
            b,
            args,
            abs_tol,
            rel_tol,
            func_calls,
            force_tmax_idx,
            subgrid_max,
            debug,
        )
        if debug:
            print("###################")
            print("## adaptive quad ##")
            print("SUCCESS: step_quad_ts from {} to {}".format(a, b))
        return tmp
    except TSIntegrationError:
        if debug:
            print(
                "FAILED: step_quad_ts from {} to {} -> SPLIT INTEGRATION".format(a, b)
            )

        return _split_quad_ts(
            f,
            a,
            b,
            args,
            abs_tol,
            rel_tol,
            cnt,
            func_calls,
            force_tmax_idx,
            subgrid_max,
            debug,
        )


def _adaptive_quad_ts(
    f,
    a,
    b,
    args,
    abs_tol,
    rel_tol,
    limit,
    func_calls,
    force_tmax_idx,
    subgrid_max,
    debug,
):
    cnt = [limit]
    try:
        res = _step_quad_ts(
            f,
            a,
            b,
            args,
            abs_tol,
            rel_tol,
            cnt,
            func_calls,
            force_tmax_idx,
            subgrid_max,
            debug,
        )
        return np.asarray([res[0], res[1], limit - cnt[0]], dtype=object)

    except TSIntegrationLimitReachedError:
        pass

    raise TSIntegrationLimitReachedError(
        "adaptive_quad_ts reach limit of {}".format(limit)
    )


def quad_osc_ts(
    f,
    a,
    b,
    periode,
    args=tuple(),
    abs_tol=1e-12,
    rel_tol=1e-12,
    limit=50,
    force_tmax_idx=None,
    adaptive=True,
    subgrid_max=0,
    get_func_calls=False,
    get_adaptive_splits=False,
    threshold=1e-12,
    osc_limit=5000,
):
    _res = np.asanyarray([0.0, 0.0, 0, 0], dtype=object)
    i = 0
    last_res_scale = 1
    while True:
        tmp_xm = a + (i + 1) * periode
        if (tmp_xm) > b:
            _res += quad_ts(
                f,
                a + i * periode,
                b,
                args,
                last_res_scale * abs_tol,
                rel_tol,
                limit,
                force_tmax_idx,
                adaptive,
                subgrid_max,
                True,
                True,
            )
            break

        new_res = quad_ts(
            f,
            a + i * periode,
            tmp_xm,
            args,
            last_res_scale * abs_tol,
            rel_tol,
            limit,
            force_tmax_idx,
            adaptive,
            subgrid_max,
            True,
            True,
        )
        _res += new_res
        # print("i", i)
        # print("new_res", new_res)
        # print("res    ", _res)
        # print("ratio  ", abs(new_res[0])/abs(_res[0]))
        last_res_scale = abs(new_res[0])

        if abs(new_res[0]) / abs(_res[0]) < threshold:
            break
        if (i > osc_limit) and (osc_limit > 0):
            raise TSOscIntegrationLimitReachedError(
                "ts_quad_osc reached the osc_limit {}".format(osc_limit)
            )
        i += 1

    res = [_res[0]]
    if get_func_calls:
        res.append(_res[2])
    if get_adaptive_splits:
        res.append(_res[3])

    if len(res) == 1:
        return res[0]
    else:
        return res
