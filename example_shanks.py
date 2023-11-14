import math

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

from tsquad import shanks
from tsquad import tsquad_py


def example_log2():
    log2 = math.log(2)

    n = 15
    ak = 0
    st = shanks.Shanks()

    for k in range(1, n):
        sk = (-1) ** (k + 1) / k
        ak += sk
        st.add_element(ak)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    axc = ax[0]
    axc.axhline(log2, color="k", ls="--", lw=1, label="$\\log(2)$")
    n = list(range(1, st.cnt + 1))
    axc.plot(n, st.data[0], ls="", marker=".", label="$a_k$")
    n = list(range(3, st.cnt + 1))
    axc.plot(n, st.data[2], ls="", marker=".", label="$\\epsilon_2(a_k)$")
    n = list(range(5, st.cnt + 1))
    axc.plot(n, st.data[4], ls="", marker=".", label="$\\epsilon_4(a_k)$")
    axc.legend()
    axc.set_xlabel("index $k$ of partial sum")
    axc.set_ylabel("value of the partial sum")

    axc = ax[1]
    n = list(range(1, st.cnt + 1))
    err = [math.fabs(s - log2) for s in st.data[0]]
    axc.plot(n, err, ls="", marker=".", label="$a_k$")
    n = list(range(3, st.cnt + 1))
    err = [math.fabs(s - log2) for s in st.data[2]]
    axc.plot(n, err, ls="", marker=".", label="$\\epsilon_2(a_k)$")
    n = list(range(5, st.cnt + 1))
    err = [math.fabs(s - log2) for s in st.data[4]]
    axc.plot(n, err, ls="", marker=".", label="$\\epsilon_4(a_k)$")

    n = [2 * i + 1 for i in range((st.cnt - 1) // 2)]
    err = [math.fabs(log2 - st.data[2 * i][0]) for i in range((st.cnt - 1) // 2)]
    axc.plot(n, err, ls="", marker="x", label="$\\epsilon_k(a_k)$", color="k")

    axc.legend()
    axc.set_xlabel("index $k$ of the partial sum")
    axc.set_ylabel("abs error of the partial sum")

    axc.set_yscale("log")

    fig.suptitle(
        "use Wynn epsilon algorithm to efficiently calculate Shanks' method\n"
        + "example: $\log(2) = \sum_{{n=1}}^\\infty (-1)^{{n+1}} / n \\approx a_k = \sum_{{n=1}}^k \\dots$"
    )

    fig.savefig("example_log2.pdf")


def example_osc_int():
    s = 0.2
    f = lambda x: 1 / x**s * math.cos(x)
    qts = tsquad_py.QuadTS(f=f)
    r_ref = math.gamma(1 - s) * math.sin(math.pi * s / 2)
    r, sht = qts._quad_osc_upper_infinite_inspect(
        a=0, period=2 * math.pi, use_mpf=False
    )

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].axhline(r_ref, color="k", ls="--", lw=1, label="analytic result")

    n = list(range(1, sht.cnt + 1))
    Ak = sht.data[0]
    ax[0].plot(n, Ak, ls="", marker=".", label="$A_k$")
    err = [abs(Aki - r_ref) for Aki in Ak]
    ax[1].plot(n, err, ls="", marker=".", label="$A_k$")

    n = [2 * i + 1 for i in range((sht.cnt - 1) // 2)]
    epsk = [sht.data[2 * i][0] for i in range((sht.cnt - 1) // 2)]

    ax[0].plot(n, epsk, ls="", marker=".", label="$\\epsilon_k(A_k)$")
    err = [abs(epski - r_ref) for epski in epsk]
    ax[1].plot(n, err, ls="", marker=".", label="$\\epsilon_k(A_k)$")

    ax[1].set_yscale("log")

    x_ticks = list(range(1, sht.cnt + 1))
    ax[0].legend()
    ax[0].set_xlabel("index $k$ of the partial sum")
    ax[0].set_ylabel("value of the partial sum")
    ax[0].set_xticks(x_ticks)

    ax[1].legend()
    ax[1].set_xlabel("index $k$ of the partial sum")
    ax[1].set_ylabel("abs error of the partial sum")
    ax[1].set_xticks(x_ticks)

    fig.suptitle(
        "use Wynn epsilon algorithm to efficiently calculate Shanks' method\n"
        + "example: $\\int_0^\\infty  \sin(x) / x^{1/5} \\mathrm{d}x = \sum_k \\int_{k\pi}^{(k+1)\pi} \dots$"
    )

    fig.savefig("example_osc_int.pdf")


def example_osc_int2():
    x1 = 5
    x2 = 15

    f_lor = lambda x: 1 / (1 + (x - x1) ** 2) + 1 / (1 + (x - x2) ** 2)
    f = lambda x: f_lor(x) * math.cos(x)

    qts = tsquad_py.QuadTS(f=f)
    r_ref = 0.3166409928992998415133346 - 0.8785518615606478863082457
    r, sht = qts._quad_osc_upper_infinite_inspect(
        a=0, period=2 * math.pi, use_mpf=False
    )

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].axhline(r_ref, color="k", ls="--", lw=1, label="analytic result")

    n = list(range(1, sht.cnt + 1))
    Ak = sht.data[0]
    ax[0].plot(n, Ak, ls="", marker=".", label="$A_k$")
    err = [abs(Aki - r_ref) for Aki in Ak]
    ax[1].plot(n, err, ls="", marker=".", label="$A_k$")

    n = [2 * i + 1 for i in range((sht.cnt - 1) // 2)]
    epsk = [sht.data[2 * i][0] for i in range((sht.cnt - 1) // 2)]

    ax[0].plot(n, epsk, ls="", marker=".", label="$\\epsilon_k(A_k)$")
    err = [abs(epski - r_ref) for epski in epsk]
    ax[1].plot(n, err, ls="", marker=".", label="$\\epsilon_k(A_k)$")

    ax[1].set_yscale("log")

    x_ticks = list(range(1, sht.cnt + 1))
    ax[0].legend()
    ax[0].set_xlabel("index $k$ of the partial sum")
    ax[0].set_ylabel("value of the partial sum")
    ax[0].set_xticks(x_ticks)

    ax[1].legend()
    ax[1].set_xlabel("index $k$ of the partial sum")
    ax[1].set_ylabel("abs error of the partial sum")
    ax[1].set_xticks(x_ticks)

    fig.suptitle(
        "use Wynn epsilon algorithm to efficiently calculate Shanks' method\n"
        + "double peak cos-integral example"
    )

    fig.savefig("example_osc_int2.pdf")


if __name__ == "__main__":
    # example_log2()
    # example_osc_int()
    example_osc_int2()
