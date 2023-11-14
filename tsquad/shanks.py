import logging

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

        logging.debug(f"init Shanks -> use_mp = {use_mp}")

        if len(a) > 0:
            self.add_sequence(a)

    def add_element(self, a):
        """
        Add new element of the series.
        Calculate the new data for the Shanks table made available by the new element.

        :param a: new element of the series
        :return: best extrapolation
        """
        logging.debug(f"add {a} to Shanks -> fill table")

        # append new element to first row
        if self.use_mp:
            self.data[0].append(mp.mpmathify(a))
        else:
            self.data[0].append(a)
        # create new row for Shanks cnt-th order
        self.data.append([])

        logging.debug("fill shank table")
        if self.cnt > 0:
            # fill Shanks table / a new anti-diagonal in the Shank matrix
            logging.debug("diff in add to row 1 -> {}".format((self.data[0][-1] - self.data[0][-2])))
            self.data[1].append(1 / (self.data[0][-1] - self.data[0][-2]))
            for i in range(1, self.cnt):
                e1 = self.data[i - 1][-2]
                e2 = self.data[i][-1] - self.data[i][-2]
                logging.debug("diff in add to row {} -> {}".format(i+1, e2))
                e = e1 + 1 / e2 if e2 != 0 else np.nan
                self.data[i + 1].append(e)
        self.cnt += 1

        logging.debug("Shanks table:")
        for i, d in enumerate(self.data):
            logging.debug(f"{i} {d}")

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

    def get_two_latest_estimates(self):
        row_idx = ((self.cnt-2) // 2)*2
        logging.debug("cnt {} col_idx {} len(data) {}".format(self.cnt, row_idx, len(self.data)))
        if self.cnt % 2 == 0:
            # even cnt -> data up to odd index
            return self.data[row_idx][-2], self.data[row_idx][-1]
        else:
            return self.data[row_idx][-1], self.data[row_idx+2][-1]

