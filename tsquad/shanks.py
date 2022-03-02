import numpy as np

class Shanks(object):
    """
    see https://doi.org/10.1016/S0377-0427(00)00355-1
    """

    def __init__(self):
        self.data = [[]]
        self.cnt = 0

    def new_element(self, a):
        self.cnt += 1
        # append new element to first row
        self.data[0].append(a)
        # create new row for Shanks cnt-th order
        self.data.append([])
        # fill Shanks table / a new anti-diagonal in the Shank matrix
        for i in range(1, self.cnt):

            self.data[i]


def dev():
    n = 10
    sh = Shanks()
    ak = 0
    for k in range(1, n):
        sk = (-1)**(k+1) / k
        ak += sk
        sh.new_element(ak)

    print(sh.data)










    print(np.log(2))


if __name__ == "__main__":
    dev()