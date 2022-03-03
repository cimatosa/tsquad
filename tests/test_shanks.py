import math
from tsquad import shanks


def test_log2():
    n = 15
    ak = 0
    st = shanks.Shanks()
    for k in range(1, n):
        sk = (-1) ** (k + 1) / k
        ak += sk
        st.add_element(ak)

    log2 = math.log(2)

    assert abs(st.data[2][0] - log2) < 1e-2
    assert abs(st.data[4][0] - log2) < 3e-4
    assert abs(st.data[6][0] - log2) < 6e-6
    assert abs(st.data[8][0] - log2) < 2e-7
