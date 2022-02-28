"""
    Numeric integration using tanh-sinh-method

    for a finite interval [a,b] as explained here:
    http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/dhb-tanh-sinh.pdf

    Note that this implementation is particularly useful when the integrand is singular at x = 0
    and the integration goes from a=0 to b. To correctly account for a singularity at a different location
    consider rewriting the integrand such that the singularity is located at 0.

    The reason is that a high precision integration needs to resolve the the integrand in quite a detail close
    to the singularity. Only for x = 0 values x + eps can be resolved for eps as small as 1e-308 (double).

    The here used method is based on the variable transformation

        x(t) = b/2(1 - g(t))
             = b/2(1 - tanh(pi/2 * sinh(t)))
             = b / 2 / (e^(pi/2*sinh(t)) cosh(pi/2 sinh(t)))

    which maps the interval [0, b] to [-inf, inf]

        int_0^b f(x) dx = int_-inf^inf f(x(t)) b/2 g'(t) dt = int_-inf^inf I(t) dt = dt sum_k f_k w_k
        g'(t) = pi cosh(t)/(2 cosh^2(pi/2 sinh(t)))

    where the singularity occurs at t = inf. It has been shown that the error can not be smaller
    than max(|I(tmin)|, |I(tmax)|). So tmin and tmax are chosen such that this limitation is
    below the desired accuracy (implemented in _get_integral_bound).
"""

import numpy as np
import os
import sys
import traceback

#  here we set some basic parameters for the TanhSinh integration routine
#  ----------------------------------------------------------------------

from . import ts_parameters
from .ts_exceptions import *

# with these parameters the nodes and weights are fixed and can therefore be precalculated
# the file named by '_fname' will hold these numbers
_fname = "ts_nodes_weights.py"

pth, fl = os.path.split(__file__)
_fname_abs = os.path.join(pth, _fname)
# if that files does not exists, it will be generated using mpmath routines.
if not os.path.exists(_fname_abs):
    print(_fname_abs, "does not exists")
    print("generate nodes and weights ...")
    from .generate_py_nodes_weights import __write_g_w_
    __write_g_w_(N0 = ts_parameters._N0,
                 k  = ts_parameters._num_sub_grids - 1,
                 fname=_fname_abs,
                 _tmax_list = ts_parameters._tmax_list,
                 _tmin = ts_parameters._tmin)
    print("done!")

from . import ts_nodes_weights

def _f_x_exception_wapper(f, x, args):
    try:
        f_x = f(x, *args)
    except Exception as e:
        sys.stderr.write(
            "calling function at x={:.8e} failed with exception {}, show traceback:\n".format(x, e.__class__.__name__))
        traceback.print_exc()
        raise TSIntegrationFunctionEvaluationError("Failed to evaluate function (Exception occurred during function call)")

    if np.isnan(f_x):
        raise TSIntegrationFunctionEvaluationError("Failed to evaluate function (function returns nan at x={:.8e})".format(x))
    return f_x

def _get_integral_bound(f, a, b, param, tol):
    """
        chose tmax such that |w_(tmax) I(g(tmax))| < tol
    """
    sc = (b-a)/2
    for i in range(ts_parameters._N_tmax):
        f_x = _f_x_exception_wapper(f, a + sc*ts_nodes_weights._1mg[i][0][-1], param)
        tmp = abs(sc * f_x * ts_nodes_weights._w[i][0][-1])
        if tmp < tol:
            break
    return i


def quad_ts(f, a, b, args = tuple(), abs_err=1e-12, rel_err=1e-12, limit=50, force_tmax_idx=None,
            adaptive=True, subgrid_max=0, get_func_calls=False, get_adaptive_splits=False, debug=False):
    """
    Integrate f(x, *args) from a to b. If a = 0 singularities at x = 0 are treated very well.

    If I is the returned value of the Integral and err_est the estimated error, the
    integration is assumed to be converged if:

        err_est < abs_err or err_est / |I| < rel_err

    Note: the integral from [a, 0] is automatically translated to -I([0, a])
    in order to better treat a potential singularity at x=0.
    """

    if (subgrid_max == 0):
        subgrid_max = ts_parameters._num_sub_grids-1
    elif subgrid_max < 3:
        print("subgrid_max ({}) set to 3".format(subgrid_max))
        subgrid_max = 3;
    elif subgrid_max > ts_parameters._num_sub_grids-1:
        print("subgrid_max ({}) must not exceed {}".format(subgrid_max, ts_parameters._num_sub_grids-1))
        raise TSIntegrationError("ts_quad integration error: subgrid_max exceeds pre-calculated nodes and weights")


    func_calls = [0]
    kwargs = {'func_calls': func_calls,
              'force_tmax_idx': force_tmax_idx,
              'subgrid_max': subgrid_max,
              'debug': debug}

    if adaptive:
        quad_method = _adaptive_quad_ts
        kwargs['limit'] = limit
    else:
        quad_method = _quad_ts

    if a == b:
        if adaptive:
            res = [0, 0, 0]
        else:
            res = [0, 0]
    elif a == -np.inf:
        if b == np.inf:
            # both inf, correct order
            res_1 = _quad_ts_minus_inf_b(quad_method, f, 0, args, abs_err, rel_err, **kwargs)
            res_2 = _quad_ts_a_plus_inf(quad_method, f, 0, args, abs_err, rel_err, **kwargs)
            res = res_1 + res_2
        else:
            # a=-inf, b is finite
            res = _quad_ts_minus_inf_b(quad_method, f, b, args, abs_err, rel_err, **kwargs)
    elif a == np.inf:
        if b == -np.inf:
            # both inf, wrong order
            res_1 = _quad_ts_minus_inf_b(quad_method, f, 0, args, abs_err, rel_err, **kwargs)
            res_2 = _quad_ts_a_plus_inf(quad_method, f, 0, args, abs_err, rel_err, **kwargs)
            res = res_1 + res_2
            res[0] *= -1
        else:
            # a=inf, b is finite
            res = _quad_ts_a_plus_inf(quad_method, f, b, args, abs_err, rel_err, **kwargs)
            res[0] *= -1
    elif b == np.inf:
        # a finite, b +inf, correct order
        res = _quad_ts_a_plus_inf(quad_method, f, a, args, abs_err, rel_err, **kwargs)
    elif b == -np.inf:
        # a finite, b - inf, wrong order, do I([-inf, a]) and return its negative
        res = _quad_ts_minus_inf_b(quad_method, f, a, args, abs_err, rel_err, **kwargs)
        res[0] *= -1
    elif b == 0:
        # map I([a, 0]) to -I([0, a]), even for a < 0
        res = quad_method(f, 0, a, args, abs_err, rel_err, **kwargs)
        res[0] *= -1
    else:
        res = quad_method(f, a, b, args, abs_err, rel_err, **kwargs)

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

def _quad_ts_a_plus_inf(quad_method, f, a, args, abs_err, rel_err, **kwargs):
    res1 = quad_method(lambda t: f(a+t, *args), 0, 1, tuple(), abs_err/2, rel_err/2, **kwargs)
    res2 = quad_method(lambda t: f(a+1/t, *args)/t**2, 0, 1, tuple(), abs_err/2, rel_err/2, **kwargs)
    return res1 + res2

def _quad_ts_minus_inf_b(quad_method, f, b, args, abs_err, rel_err, **kwargs):
    res1 = quad_method(lambda t: f(b-t, *args), 0, 1, tuple(), abs_err/2, rel_err/2, **kwargs)
    res2 = quad_method(lambda t: f(b-1/t, *args)/t**2, 0, 1, tuple(), abs_err/2, rel_err/2, **kwargs)
    return res1 + res2


def _quad_ts(f, a, b, args, abs_err, rel_err, func_calls, force_tmax_idx, subgrid_max, debug=False):
    """
        performs the actual TanhSinh integration
    """
    if force_tmax_idx is None:
        tmax_idx = _get_integral_bound(f, a, b, args, abs_err)
        func_calls[0] += (tmax_idx+1)
    else:
        tmax_idx = force_tmax_idx

    data_dt = ts_nodes_weights.dt[tmax_idx]
    data_1mg = ts_nodes_weights._1mg[tmax_idx]
    data_w = ts_nodes_weights._w[tmax_idx]

    eps = 10**-14

    I_res_n2 = 0
    I_res_n1 = 0
    I_res = 0
    sum_tmp = 0

    sc = (b-a)/2

    f_x_max = _f_x_exception_wapper(f, a + sc*data_1mg[0][+0], args)
    w_f_tmax = sc * f_x_max * data_w[0][0]
    f_x_min = _f_x_exception_wapper(f, a + sc*data_1mg[0][-1], args)
    w_f_tmin = sc * f_x_min * data_w[0][-1]

    func_calls[0] += 2

    d4_tmin = abs(w_f_tmin)
    d4_tmax = abs(w_f_tmax)
    err4 = max(d4_tmin, d4_tmax)


    err_est = np.nan
    err1 = err2 = err3 = np.nan
    N = subgrid_max

    assert N >= 2

    if debug:
        print('##  TS integration in debug mode ##')
        print("## " + '-'*29 + ' ##')
        print("tmin", a + sc * data_1mg[0][-1])
        print("tmax", a + sc * data_1mg[0][+0])
        print("f(t_min)", f_x_min)
        print("f(t_max)", f_x_max)
        print("d4_tmin", d4_tmin)
        print("d4_tmax", d4_tmax)
        print("## " + '-' * 29 + ' ##')

    success = False
    may_be_success = False
    I_res_final = 0
    err_est_final = 0

    for n in range(N+1):
        max_tmp = 0
        for k in range(len(data_w[n])):
            f_x = _f_x_exception_wapper(f, a+ sc*data_1mg[n][k], args)
            w_f = sc * f_x * data_w[n][k]

            max_tmp = max(max_tmp, abs(w_f))
            sum_tmp += w_f
        func_calls[0] += len(data_w[n])
        I_res_n2 = I_res_n1
        I_res_n1 = I_res
        I_res = sum_tmp*data_dt[n]
        err3 = abs(eps * max_tmp)

        if debug:
            print("n", n, "I_n", I_res)

        if n >= 2:
            d1 = abs(I_res - I_res_n1)
            if debug:
                print("d1 = I_n - I_(n-1)    {:.8e} -> err {:.16e}".format(d1, d1**2))
            if (d1 == 0):
                if debug:
                    print("best we can have!")
                    print("return, {:.16e} +- {:.4e}".format(I_res, err_est))
                err_est = max(err3, err4)
                return np.asarray([I_res, err_est])
            else:
                d1_log = np.log10(d1)

            d2 = abs(I_res - I_res_n2)
            if debug:
                print("d2 = I_n - I_(n-2)    {:.8e}".format(d2))

            err1 = d1**2
            if debug:
                print("err1 = {:.8e}".format(err1))

            if (d2 > 1e-308):
                d2_log = np.log10(d2)
                tmp = (d1_log ** 2 / d2_log)
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

            if (err_est < abs_err) or (err_est < rel_err*abs(I_res)):
                # the error constrain is fulfilled
                if may_be_success:
                    # the error constrain has been fulfilled twice in a row
                    # so we can quit here

                    # we set success and save the current results
                    # only if we have not already succeeded
                    # (might be the case in debug mode to see convergence properties)
                    if not success:
                        success = True
                        I_res_final = I_res
                        err_est_final = err_est
                    # is not in debug mode exit loop here
                    if not debug:
                        break
                    else:
                        print("err constrain fulfilled second time in a row, BUT keep in looping due to debug mode")
                else:
                    # first time we fulfilled error constrain
                    may_be_success = True
                    if debug:
                        print("err contrains fulfilled first time (may_be_success set to True)")


            else:
                # when ever the error constrain is not fulfilled
                # we reset may_be_success
                may_be_success = False
                if debug:
                    print("err contrains NOT fulfilled (may_be_success set to False)")


    if success:
        if debug:
            print("return, {:.16e} +- {:.4e}".format(I_res_final, err_est_final))
        return np.asarray([I_res_final, err_est_final], dtype=object)


    raise TSIntegrationError("Required tolerance not achieved!\n"+
                             "err_est       = {} > abs_err = {}\n".format(err_est, abs_err)+
                             "err_est/I_res = {} > rel_err = {}\n".format(err_est/I_res, abs_err) +
                             "Reached max number N={} of sub grids\n".format(N)+
                             "tmax: {}\n".format(ts_parameters._tmax_list[tmax_idx])+
                             "err: d1^2/d2: {}\n".format(err2)+
                             "err:     2d1: {}\n".format(err1)+
                             "err:      d3: {}\n".format(err3)+
                             "err:      d4: {}".format(err4))



def _split_quad_ts(f, a, b, args, abs_err, rel_err, cnt, func_calls, force_tmax_idx, subgrid_max, debug):
    res1 = _step_quad_ts(f, a, (a+b)/2, args, abs_err, rel_err, cnt, func_calls, force_tmax_idx, subgrid_max, debug)
    res2 = _step_quad_ts(f, (a+b)/2, b, args, abs_err, rel_err, cnt, func_calls, force_tmax_idx, subgrid_max, debug)
    return res1 + res2

def _step_quad_ts(f, a, b, args, abs_err, rel_err, cnt, func_calls, force_tmax_idx, subgrid_max, debug):
    cnt[0] = cnt[0] - 1
    if cnt[0] == 0:
        raise TSIntegrationLimitReachedError()
    try:
        tmp = _quad_ts(f, a, b, args, abs_err, rel_err, func_calls, force_tmax_idx, subgrid_max, debug)
        if debug:
            print('###################')
            print("## adaptive quad ##")
            print("SUCCESS: step_quad_ts from {} to {}".format(a, b))
        return tmp
    except TSIntegrationError:
        if debug:
            print("FAILED: step_quad_ts from {} to {} -> SPLIT INTEGRATION".format(a, b))

        return _split_quad_ts(f, a, b, args, abs_err, rel_err, cnt, func_calls, force_tmax_idx, subgrid_max, debug)

def _adaptive_quad_ts(f, a, b, args, abs_err, rel_err, limit, func_calls, force_tmax_idx, subgrid_max, debug):
    cnt = [limit]
    try:
        res = _step_quad_ts(f, a, b, args, abs_err, rel_err, cnt, func_calls, force_tmax_idx, subgrid_max, debug)
        return np.asarray([res[0], res[1], limit - cnt[0]], dtype=object)

    except TSIntegrationLimitReachedError:
        pass

    raise TSIntegrationLimitReachedError("adaptive_quad_ts reach limit of {}".format(limit))

def quad_osc_ts(f, a, b, periode, args=tuple(), abs_err=1e-12, rel_err=1e-12, limit=50, force_tmax_idx=None,
                adaptive=True, subgrid_max=0, get_func_calls=False, get_adaptive_splits=False,
                threshold=1e-12, osc_limit=5000):
    _res = np.asanyarray([0.,0.,0,0], dtype=object)
    i = 0
    last_res_scale = 1
    while True:
        tmp_xm = a+(i+1)*periode
        if (tmp_xm) > b:
            _res += quad_ts(f, a+i*periode, b, args, last_res_scale*abs_err, rel_err, limit, force_tmax_idx,
                           adaptive, subgrid_max, True, True)
            break

        new_res = quad_ts(f, a+i*periode, tmp_xm, args, last_res_scale*abs_err, rel_err, limit,
                          force_tmax_idx, adaptive, subgrid_max, True, True)
        _res += new_res
        # print("i", i)
        # print("new_res", new_res)
        # print("res    ", _res)
        # print("ratio  ", abs(new_res[0])/abs(_res[0]))
        last_res_scale = abs(new_res[0])

        if (abs(new_res[0])/abs(_res[0]) < threshold):
            break
        if (i > osc_limit) and (osc_limit > 0):
            raise TSOscIntegrationLimitReachedError("ts_quad_osc reached the osc_limit {}".format(osc_limit))
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

