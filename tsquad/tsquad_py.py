"""
    Numeric Integration Using Tanh-Sinh Variable Transformation

    This is the main module which implements the Tanh-Sinh method for numeric integration.
"""

# python import
import logging
import math
import traceback
import typing

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

    if math.isnan(f_x):
        raise TSIntegrationFunctionEvaluationError(
            "Failed to evaluate function (function returns nan at x={:.8e})".format(x)
        )
    return f_x


class QuadRes(object):
    __slots__ = ("I", "err", "func_calls", "rec_steps")

    def __init__(self, I=0, err=0, func_calls=0, rec_steps=0):
        self.I = I
        self.err = err
        self.func_calls = func_calls
        self.rec_steps = rec_steps

    def __add__(self, other):
        r = QuadRes(
            I=self.I + other.I,
            err=self.err + other.err,
            func_calls=self.func_calls + other.func_calls,
            rec_steps=self.rec_steps + other.rec_steps,
        )
        return r

    def __str__(self):
        return "QuadRes(I={}, err={}, func_calls={}, rec_steps={})".format(
            self.I, self.err, self.func_calls, self.rec_steps
        )


class QuadTS(object):
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
    :param args: arguments passed to `f`
    :param abs_tol: absolute tolerance
    :param rel_tol: relative tolerance
    :param recursive: if True, allow for recursive application of the tanh-sinh scheme
    :param rec_limit: maximum number of recursions allowed
    :param force_t_max_idx: Set the upper boundary/truncation for the (transformed) t-integral by hand,
                            this controls the resolution of the lower bound a of the x-integral.
                            Mainly needed by testing purposes.
    :param subgrid_max: Set the number of sub-grids to use. If `subgrid_max=0` use the largest number ava
    :param debug: if True, enable debug messages
    """

    def __init__(
        self,
        f: typing.Callable,
        args: tuple = tuple(),
        abs_tol: float = 1e-12,
        rel_tol: float = 1e-12,
        recursive: bool = True,
        rec_limit: int = 50,
        force_t_max_idx: [None, int] = None,
        subgrid_max=0,
        debug=False,
    ):
        # init class members
        self.f = f
        self.args = args
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.recursive = recursive
        self.rec_limit = rec_limit
        self.force_t_max_idx = force_t_max_idx
        self.subgrid_max = subgrid_max
        self.debug = debug

        # process data
        if self.subgrid_max == 0:
            self.subgrid_max = tsconfig.num_sub_grids - 1
        elif self.subgrid_max < 3:
            logging.info("subgrid_max ({}) set to 3".format(subgrid_max))
            self.subgrid_max = 3
        elif subgrid_max > tsconfig.num_sub_grids - 1:
            raise ValueError("subgrid_max exceeds pre-calculated nodes and weights")

    def _get_integral_bound(self, a, b):
        """
            chose t_max such that |w_(t_max) I(g(t_max))| < abs_tol
        """
        sc = (b - a) / 2
        for i in range(tsconfig.N_t_max):
            f_x = _f_x_exception_wrapper(
                self.f, a + sc * nodes_weights._1mg[i][0][-1], self.args
            )
            tmp = abs(sc * f_x * nodes_weights._w[i][0][-1])
            if tmp < self.abs_tol:
                return i

    ####################################################################################################################
    ##  simple (single run) Tanh-Sinh integration scheme, i.e., the core routine
    ####################################################################################################################

    def _quad(self, a, b) -> QuadRes:
        """
        helper function that performs the actual Tanh-Sinh integration and estimates the error
        (http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/dhb-tanh-sinh.pdf.)

        Perform the numeric integration of int_a^b f(x, *args) dx.
        Sequentially doubles the number of nodes until die desired accuracy is reached.
        If the maximum number of doubling (given by subgrid_max) is reached without achieving
        the desired accuracy, and TSIntegrationError is raises.

        :param a: lower integral boundary
        :param b: upper integral boundary
        :return: a QuadRes result object where `I` contains the value of the numeric integration and `err` the estimate
        of the error. In addition, `func_calls` the `adaptive_splits` is provided by the result object.
        """

        local_func_calls = 0

        if self.force_t_max_idx is None:
            t_max_idx = self._get_integral_bound(a, b)
            local_func_calls += t_max_idx + 1
        else:
            t_max_idx = self.force_t_max_idx

        data_dt = nodes_weights.dt[t_max_idx]
        data_1mg = nodes_weights._1mg[t_max_idx]
        data_w = nodes_weights._w[t_max_idx]

        eps = 10 ** -14

        I_res_n2 = 0
        I_res_n1 = 0
        I_res = 0
        sum_tmp = 0

        sc = (b - a) / 2

        f_x_max = _f_x_exception_wrapper(self.f, a + sc * data_1mg[0][+0], self.args)
        w_f_t_max = sc * f_x_max * data_w[0][0]
        f_x_min = _f_x_exception_wrapper(self.f, a + sc * data_1mg[0][-1], self.args)
        w_f_t_min = sc * f_x_min * data_w[0][-1]

        local_func_calls += 2

        d4_t_min = abs(w_f_t_min)
        d4_t_max = abs(w_f_t_max)
        err4 = max(d4_t_min, d4_t_max)

        err_est = math.nan
        err1 = err2 = err3 = math.nan
        N = self.subgrid_max
        assert N >= 2

        if self.debug:
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
                f_x = _f_x_exception_wrapper(self.f, a + sc * data_1mg[n][k], self.args)
                w_f = sc * f_x * data_w[n][k]

                max_tmp = max(max_tmp, abs(w_f))
                sum_tmp += w_f
            local_func_calls += len(data_w[n])
            I_res_n2 = I_res_n1
            I_res_n1 = I_res
            I_res = sum_tmp * data_dt[n]
            err3 = abs(eps * max_tmp)

            if self.debug:
                print("n", n, "I_n", I_res)

            if n >= 2:
                d1 = abs(I_res - I_res_n1)
                if self.debug:
                    print(
                        "d1 = I_n - I_(n-1)    {:.8e} -> err {:.16e}".format(
                            d1, d1 ** 2
                        )
                    )
                if d1 == 0:
                    if self.debug:
                        print("best we can have!")
                        print("return, {:.16e} +- {:.4e}".format(I_res, err_est))
                    err_est = max(err3, err4)
                    return QuadRes(
                        I=I_res, err=err_est, func_calls=local_func_calls, rec_steps=1,
                    )
                else:
                    d1_log = math.log10(d1)

                d2 = abs(I_res - I_res_n2)
                if self.debug:
                    print("d2 = I_n - I_(n-2)    {:.8e}".format(d2))

                err1 = d1 ** 2
                if self.debug:
                    print("err1 = {:.8e}".format(err1))

                if d2 > 1e-308:
                    d2_log = math.log10(d2)
                    tmp = d1_log ** 2 / d2_log
                    if self.debug:
                        print("d1_log", d1_log)
                        print("d2_log", d2_log)
                        print("tmp   ", tmp)
                    if tmp < -308:
                        err2 = 0
                        if self.debug:
                            print("err2 = 0  (due to 10 ** d1log^2/d2log underflow")
                    elif tmp > 308:
                        err2 = 10
                        if self.debug:
                            print("err2 = 10  (due to 10 ** d1log^2/d2log overflow")
                    else:
                        err2 = 10 ** tmp
                        if self.debug:
                            print("err2 = {:.8e}".format(err2))
                else:
                    err2 = 10
                    if self.debug:
                        print("err2 = 10  (due to d2 < 1e-308)")

                if self.debug:
                    print("err3 = {:.8e}".format(err3))

                if err2 >= 10:
                    if self.debug:
                        print("err1 >= 10  ->  continue")
                    may_be_success = False
                    continue

                err_est = max(err1, err2, err3, err4)

                if (err_est < self.abs_tol) or (err_est < self.rel_tol * abs(I_res)):
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
                        if not self.debug:
                            break
                        else:
                            print(
                                "err constrain fulfilled second time in a row, BUT keep in looping due to debug mode"
                            )
                    else:
                        # first time we fulfilled error constrain
                        may_be_success = True
                        if self.debug:
                            print(
                                "err constrains fulfilled first time (may_be_success set to True)"
                            )
                else:
                    # when ever the error constrain is not fulfilled
                    # we reset may_be_success
                    may_be_success = False
                    if self.debug:
                        print(
                            "err constrains NOT fulfilled (may_be_success set to False)"
                        )

        if success:
            if self.debug:
                print("return, {:.16e} +- {:.4e}".format(I_res_final, err_est_final))
            return QuadRes(
                I=I_res_final,
                err=err_est_final,
                func_calls=local_func_calls,
                rec_steps=1,
            )

        raise TSIntegrationError(
            "Required tolerance not achieved!\n"
            + "err_est       = {} > abs_tol = {}\n".format(err_est, self.abs_tol)
            + "err_est/I_res = {} > rel_tol = {}\n".format(
                err_est / I_res, self.abs_tol
            )
            + "Reached max number N={} of sub grids\n".format(N)
            + "tmax: {}\n".format(tsconfig.t_max_list[t_max_idx])
            + "err: d1^2/d2: {}\n".format(err2)
            + "err:     2d1: {}\n".format(err1)
            + "err:      d3: {}\n".format(err3)
            + "err:      d4: {}".format(err4)
        )

    ####################################################################################################################
    ##  adaptive Tanh-Sinh integration scheme, i.e., split integral if needed
    ##  perform simple Tanh-Sinh integration on sub-intervals
    ####################################################################################################################

    def _split(self, a: numeric, b: numeric, limit: int) -> QuadRes:
        """
        Split the interval [a,b] in the middle and perform an integration step on each sub-interval.
        :return: the sum of both sub-results as QuadRes
        """
        res1 = self._step(a, (a + b) / 2, limit - 1)
        res2 = self._step((a + b) / 2, b, limit - res1.rec_steps)
        # this adds I, err, func_calls and adaptive_splits
        return res1 + res2

    def _step(self, a: numeric, b: numeric, limit: int) -> QuadRes:
        """
        Integrate from a to b, if accuracy is not reaches, split the interval.
        :return: the result as QuadRes
        """
        if limit == 0:
            raise TSIntegrationLimitReachedError(
                "recursive integration has reached its limit of {}".format(
                    self.rec_limit
                )
            )
        try:
            tmp_res = self._quad(a, b)
            if self.debug:
                print("###################")
                print("## adaptive quad ##")
                print("SUCCESS: step_quad_ts from {} to {}".format(a, b))
            return tmp_res
        except TSIntegrationError:
            if self.debug:
                print(
                    "FAILED: step_quad_ts from {} to {} -> SPLIT INTEGRATION".format(
                        a, b
                    )
                )
            return self._split(a, b, limit)

    ####################################################################################################################
    ##  high level functions
    ####################################################################################################################

    def recursive_quad(self, a: numeric, b: numeric) -> QuadRes:
        """
        Perform recursive integration.
        :param a: lower bound
        :param b: upper bound
        :return: result as QuadRes
        """
        return self._step(a, b, self.rec_limit)

    def simple_quad(self, a: numeric, b: numeric) -> QuadRes:
        """
        Perform simple (non-recursive) integration
        :param a: lower bound
        :param b: upper bound
        :return: result as QuadRes
        """
        return self._quad(a, b)

    def quad_finite_boundary(self, a: numeric, b: numeric) -> QuadRes:
        """
        Integrate, choose method based on class initialization
        :return: result ad QuadRes
        """
        if self.recursive:
            return self.recursive_quad(a, b)
        else:
            return self.simple_quad(a, b)

    ####################################################################################################################
    ##  treat improper integral (unbound integration interval) by a mypping to a finite region
    ####################################################################################################################

    def quad_upper_infinite(self, a: numeric) -> QuadRes:
        """
        Integrate to infinity by splitting into [a, a+1] and [a+1, inf].
        In that way a potential singularity at a (=0) is treated correctly.
        The second interval is than mapped to [0, 1] where, due to the variable transformation
        t = 1/(x-a), again, a singularity at t=0 appears:

            int_(a+1)^inf f(x) dx = int_0^1 f(1/t + a) / t**2 .

        Each interval is treated independently, which means that for both intervals the same recursion limit
        is used, therefore the recursion limit is effectively doubled.
        :param a: lower bound
        :return: result as QuadRes
        """
        res1 = self.quad_finite_boundary(a, a + 1)

        tsq2 = QuadTS(
            f=lambda t: self.f(1 / t + a, *self.args) / t ** 2,
            args=tuple(),
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            recursive=self.recursive,
            rec_limit=self.rec_limit,
            force_t_max_idx=self.force_t_max_idx,
            subgrid_max=self.subgrid_max,
            debug=self.debug,
        )
        res2 = tsq2.quad_finite_boundary(0, 1)

        return res1 + res2

    def quad_lower_infinite(self, b: numeric) -> QuadRes:
        """
            As in `quad_upper_infinite` split into [-inf, b-1] and [b-1, b].
            For the first interval use t = -1/(x-b) which yields

                int_-inf^(b-1) f(x) dx = int_0^1 f(-1/t + b)/t**2 dt

            Each interval is treated independently, which means that for both intervals the same recursion limit
            is used, therefore the recursion limit is effectively doubled.
            :param b: lower bound
            :return: result as QuadRes
        """

        res1 = self.quad_finite_boundary(b - 1, b)

        tsq2 = QuadTS(
            f=lambda t: self.f(-1 / t + b, *self.args) / t ** 2,
            args=tuple(),
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            recursive=self.recursive,
            rec_limit=self.rec_limit,
            force_t_max_idx=self.force_t_max_idx,
            subgrid_max=self.subgrid_max,
            debug=self.debug,
        )
        res2 = tsq2.quad_finite_boundary(0, 1)

        return res1 + res2

    def quad(self, a: numeric, b: numeric):
        """
        General method used to integrate from a to b.
        Automatically handly infinite boundaries.
        :param a: lower boundary
        :param b: upper boundary
        :return: the result as QuadRes
        """

        if a == b:
            return QuadRes(I=0, err=0, func_calls=0, rec_steps=0)

        if b < a:
            c = a
            a = b
            b = c
            sign = -1
        else:
            sign = +1

        if a == -math.inf:
            if b == math.inf:
                # both inf, correct order
                res_1 = self.quad_lower_infinite(0)
                res_2 = self.quad_upper_infinite(0)
                res = res_1 + res_2
            else:
                # a=-inf, b is finite
                res = self.quad_lower_infinite(b)
        else:
            if b == math.inf:
                # a is finite, b=inf
                res = self.quad_upper_infinite(a)
            else:
                # both finite
                res = self.quad_finite_boundary(a, b)

        res.I *= sign
        return res

# https://github.com/pjlohr/WynnEpsilon/blob/master/wynnpi.py
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
