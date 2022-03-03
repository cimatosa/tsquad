import math
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

from tsquad import shanks


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
        "use Wynn epsilon algorithm to efficiently calculate Shanks' method\nexample: $\log(2) = \sum_{{n=1}}^\\infty (-1)^{{n+1}} / n \\approx a_k = \sum_{{n=1}}^k \\dots$"
    )

    plt.show()


if __name__ == "__main__":
    example_log2()
