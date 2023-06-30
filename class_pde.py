"""
Created on Mon 26 June 2023

@author: effiehan

import numpy as np
from scipy import interpolate

from pricingmethod.class_para import parameters


# needs to be changed #################################################

class snowball_pde(parameters):

    def __init__(self):
        parameters.__init__(self)
        self.Mat_Inv = self.__getInvMat()  # inverse matrix used in backwardation
        self.option_price = np.nan
        self.__V = np.zeros((self.J + 1, self.N + 1))  # backwardation grid
        self.delta = np.nan
        self.gamma = np.nan
        self.vega = np.nan

    """" 3 helper function to calculate inverse matrix needed"""

    def __a0(self, x):
        return 0.5 * self.dt * ((self.r - self.div - self.repo) * x - self.v ** 2 * x ** 2)

    def __a1(self, x):
        return 1 + self.r * self.dt + self.v ** 2 * x ** 2 * self.dt

    def __a2(self, x):
        return 0.5 * self.dt * (-(self.r - self.div - self.repo) * x - self.v ** 2 * x ** 2)

    def __getInvMat(self):
        """Calculating Inverse Matrix"""

        # first line
        V = np.zeros((self.J + 1, self.J + 1))
        V[0, 0] = 1.0 / (1 - self.r * self.dt)

        # lines between
        for i in range(1, self.J):
            V[i, i - 1] = self.__a0(i)
            V[i, i] = self.__a1(i)
            V[i, i + 1] = self.__a2(i)

        # last line
        V[self.J, self.J - 1] = self.__a0(self.J) - self.__a2(self.J)
        V[self.J, self.J] = self.__a1(self.J) + 2 * self.__a2(self.J)

        return np.matrix(V).I

    def __interpolate_price(self, y, s):

        x = [self.lb + self.dS * i for i in range(self.J + 1)]
        f = interpolate.interp1d(x, y, kind='cubic')

        return float(f(s))

    def __compute_autocall(self):
        """present value of autocall coupon if KO"""

        # initialize payoff at maturity
        V_terminal = np.zeros((self.J + 1, 1))
        V_terminal[slice(int((self.KO_Barrier - self.lb) / self.dS), \
                         self.J + 1, 1)] = self.KO_Coupon + 1
        V_matrix = np.zeros((self.J + 1, self.N + 1))
        V_matrix[:, -1] = V_terminal.reshape((self.J + 1,))

        # backwardation
        for i in range(self.M):
            for j in range(self.n):
                idx = i * self.n + j
                V_matrix[:, self.N - idx - 1] = (self.Mat_Inv * \
                                                 V_matrix[:, self.N - idx].reshape((self.J + 1, 1))).reshape(
                    (self.J + 1,))

            # pay coupon if KO at the end of each month
            KO_Coupon_temp = self.KO_Coupon * (self.T * 12 - i - 1) / 12
            if i != self.M - 1:
                V_matrix[:, self.N - idx - 1] \
                    [slice(int((self.KO_Barrier - self.lb) / self.dS), self.J + 1, 1)] = KO_Coupon_temp + 1

        self.__V = self.__V + V_matrix

    def __compute_bonus(self):
        """present value of bonus coupon if not KO and not KI"""

        # initialize payoff at maturity
        V_terminal = np.zeros((self.J + 1, 1))
        V_terminal[slice(int((self.KI_Barrier - self.lb) / self.dS), \
                         int((self.KO_Barrier - self.lb) / self.dS), 1)] = self.KO_Coupon + 1
        V_matrix = np.zeros((self.J + 1, self.N + 1))
        V_matrix[:, -1] = V_terminal.reshape((self.J + 1,))

        # backwardation
        for i in range(self.M):
            for j in range(self.n):
                idx = i * self.n + j
                V_matrix[:, self.N - idx - 1] = (self.Mat_Inv * \
                                                 V_matrix[:, self.N - idx].reshape((self.J + 1, 1))).reshape(
                    (self.J + 1,))

                # no bonus coupon if knock in, observed daily
                V_matrix[:, self.N - idx - 1][slice(0, \
                                                    int((self.KI_Barrier - self.lb) / self.dS), 1)] = 0

            # no bonus coupon if knock out, observed monthly
            if i != self.M - 1:
                V_matrix[:, self.N - idx - 1] \
                    [slice(int((self.KO_Barrier - self.lb) / self.dS), self.J + 1, 1)] = 0

        self.__V = self.__V + V_matrix

    def __compute_put_UO(self):
        """value of put up and out"""

        # initialize payoff at maturity
        V_terminal = np.array([-1 + max(self.K - i * self.dS, 0) for i in range(self.J + 1)]).reshape((self.J + 1, 1))
        V_terminal[slice(int((self.KO_Barrier - self.lb) / self.dS) + 0, self.J + 1, 1)] = 0
        V_matrix = np.zeros((self.J + 1, self.N + 1))
        V_matrix[:, -1] = V_terminal.reshape((self.J + 1,))

        # backwardation
        for i in range(self.M):
            for j in range(self.n):
                idx = i * self.n + j
                V_matrix[:, self.N - idx - 1] = (self.Mat_Inv * \
                                                 V_matrix[:, self.N - idx].reshape((self.J + 1, 1))).reshape(
                    (self.J + 1,))

            # nothing if Knock out, observed monthly
            if i != self.M - 1:
                V_matrix[:, self.N - idx - 1] \
                    [slice(int((self.KO_Barrier - self.lb) / self.dS) + 0, self.J + 1, 1)] = 0

        self.__V = self.__V - V_matrix

    def __compute_put_UO_DO(self):
        """value of put down&out and up&out"""

        # initialize payoff at maturity
        V_terminal = np.array([-1 + max(self.K - i * self.dS, 0) for i in range(self.J + 1)]).reshape((self.J + 1, 1))
        V_terminal[slice(0, int((self.KI_Barrier - self.lb) / self.dS), 1)] = 0
        V_terminal[slice(int((self.KO_Barrier - self.lb) / self.dS) + 1, self.J + 1, 1)] = 0
        V_matrix = np.zeros((self.J + 1, self.N + 1))
        V_matrix[:, -1] = V_terminal.reshape((self.J + 1,))

        # backwardation
        for i in range(self.M):
            for j in range(self.n):
                idx = i * self.n + j
                V_matrix[:, self.N - idx - 1] = (self.Mat_Inv * \
                                                 V_matrix[:, self.N - idx].reshape((self.J + 1, 1))).reshape(
                    (self.J + 1,))

                # nothing if knock in, observed daily
                V_matrix[:, self.N - idx - 1] \
                    [slice(0, int((self.KI_Barrier - self.lb) / self.dS), 1)] = 0

            # nothing if knock out, observed monthly
            if i != self.M - 1:
                V_matrix[:, self.N - idx - 1] \
                    [slice(int((self.KO_Barrier - self.lb) / self.dS) + 1, self.J + 1, 1)] = 0

        self.__V = self.__V + V_matrix

    def compute_price(self):
        """compute the price of snowball option"""

        # reset inverse Matrix in case vol has been changed
        self.Mat_Inv = self.__getInvMat()

        self.__V = np.zeros((self.J + 1, self.N + 1))
        self.__compute_autocall()
        self.__compute_bonus()
        self.__compute_put_UO()
        self.__compute_put_UO_DO()

        self.option_price = self.__interpolate_price(self.__V[:, 0], self.S)

    def compute_greeks(self):
        """"compute greeks of snowball option"""

        epsilon = 0.01
        self.v = self.v + epsilon
        self.compute_price()
        P3 = self.option_price

        # back to original and price
        self.v = self.v - epsilon
        self.compute_price()
        P0 = self.option_price
        P1 = self.__interpolate_price(self.__V[:, 0], self.S * (1 - epsilon))
        P2 = self.__interpolate_price(self.__V[:, 0], self.S * (1 + epsilon))

        self.delta = (P2 - P1) / (2 * self.S * epsilon)
        self.gamma = (P1 + P2 - 2 * P0) / (self.S ** 2 * epsilon ** 2)
        self.vega = (P3 - P0) / epsilon

    def bisection_method(self, lower_bound, upper_bound, tolerance, price_tolerance=1e-6):
        mid_point = (lower_bound + upper_bound) / 2.0
        while (upper_bound - lower_bound) / 2.0 > tolerance:
            self.KO_Coupon = mid_point
            self.compute_price()
            if abs(self.option_price) < price_tolerance:
                return mid_point
            elif self.option_price > 0:
                upper_bound = mid_point
            else:
                lower_bound = mid_point
            mid_point = (lower_bound + upper_bound) / 2.0
        return mid_point


if __name__ == '__main__':
    import time

    pde = snowball_pde()
    # pde.print_parameters()
    tic = time.time()
    print("Starting calculating......", end="")
    pde.compute_price()
    print("Done.")
    print("Option price = ", pde.option_price)
    print("Running time = ", time.time() - tic, "s")
    print("---------------------------------------------")

    tic = time.time()
    print("Calculating Greeks.....", end="")
    pde.compute_greeks()
    print("Done.")
    print("Option delta = ", pde.delta)
    print("Option gamma = ", pde.gamma)
    print("Option vega = ", pde.vega)
    print("Running time = ", time.time() - tic, "s")
    print("---------------------------------------------")
    # Choose your initial bounds and tolerance
    lower_bound = 0.01  # your chosen lower bound for KO_Coupon
    upper_bound = 0.1  # your chosen upper bound for KO_Coupon
    tolerance = 1e-6  # your chosen tolerance for the bisection method

    # Call the bisection method on your object
    coupon_rate = pde.bisection_method(lower_bound, upper_bound, tolerance)

    # Now coupon_rate should hold the value of KO_Coupon for which the option price is approximately zero
    print("Coupon rate by pde method: ", coupon_rate)
