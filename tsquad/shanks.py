import numpy as np
import mpmath as mp


class Shanks(object):
    """
    see https://doi.org/10.1016/S0377-0427(00)00355-1
    """

    def __init__(self, a=[], use_mp=False):
        """
        Init with the sequence a.
        If `use_mp` is True, use mpmath float/complex for the Shanks table. Specify the
        precision before the calculation.
        """
        self.data = [[]]
        self.cnt = 0
        self.use_mp = use_mp

        if len(a) > 0:
            self.add_sequence(a)

    def add_element(self, a):
        """
        Add new element of the series.
        Calculate the new data for the Shanks table made available by the new element.

        :param a: new element of the series
        :return: best extrapolation
        """

        # append new element to first row
        if self.use_mp:
            self.data[0].append(mp.mpmathify(a))
        else:
            self.data[0].append(a)
        # create new row for Shanks cnt-th order
        self.data.append([])

        if self.cnt > 0:
            # fill Shanks table / a new anti-diagonal in the Shank matrix
            self.data[1].append(1 / (self.data[0][-1] - self.data[0][-2]))
            for i in range(1, self.cnt):
                e1 = self.data[i - 1][-2]
                e2 = self.data[i][-1] - self.data[i][-2]
                e = e1 + 1 / e2
                self.data[i + 1].append(e)
        self.cnt += 1

        return self.get_shanks()

    def add_sequence(self, a):
        """
        Add a sequence of new elements.
        :param a: sequence of elements
        :return: best extrapolation after adding
        """
        for ai in a:
            r = self.add_element(ai)
        return r

    def get_shanks(self, k=-1):
        """
        Return the latest element of the k-th order Shanks transformation.
        :param k: order (= multiple applications of the Shanks transform),
                  use negative value to access the highest (-1), the second highest (-2) etc. order
        :return: best current extrapolation
        """
        if k < 0:
            k_max = (self.cnt - 1) // 2
            k = k_max + k + 1
        i = 2 * k

        r = self.data[i][-1]
        if self.use_mp:
            if isinstance(r, mp.mpc):
                r = complex(r)
            else:
                r = float(r)

        return r
