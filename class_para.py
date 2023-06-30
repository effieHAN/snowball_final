# -*- coding: utf-8 -*-


class parameters:
    """parameters to used for pricing snowball option using monte carlo"""
    def __init__(self):
        """initialize parameters"""

        self.S = 1  # underlying spot
        self.K = 1  # strike
        self.KI_Barrier = 0.85  # down in barrier of put
        self.KO_Barrier = 1.03  # autocall barrier
        self.KO_Coupon = 0.2  # autocall coupon (p.a.)
        # self.Bonus_Coupon = 0.22  # bonus coupon (p.a.)
        self.r = 0.03  # risk-free interest rate
        self.div = 0  # 0.01  # dividend rate
        self.repo = 0  # 0.08  # repo rate
        self.T = 1  # time to maturity in years
        self.v = 0.13  # volatility
        self.N = 252 * self.T  # number of discrete time points for whole tenor
        self.n = int(self.N / (self.T * 12))  # number of dicrete time point for each month
        self.M = int(self.T * 12)  # number of months
        self.dt = self.T / self.N  # delta t
        self.simulations = int(3e5)
        self.lockmonth= 0
        self.J = 900  # number of steps of uly in the scheme
        self.lb = 0  # lower bound of domain of uly
        self.ub = 1.5  # upper bound of domain of uly
        self.dS = (self.ub - self.lb) / self.J  # delta S

    def print_parameters(self):
        """print parameters"""

        print("---------------------------------------------")
        print("Pricing a Snowball option")
        print("---------------------------------------------")
        print("Parameters of Snowball Option Pricer:")
        print("---------------------------------------------")
        print("Underlying Asset Price = ", self.S)
        # print("Strike = ", self.K)
        print("Knock-in Barrier = ", self.KI_Barrier)
        print("Autocall Barrier = ", self.KO_Barrier)
        print("Autocall Coupon = ", self.KO_Coupon)
        # print("Bonus Coupon = ", self.Bonus_Coupon)
        print("Risk-Free Rate =", self.r)
        # print("Dividend Rate =", self.div)
        # print("Repo Rate =", self.repo)
        print("Years Until Expiration = ", self.T)
        print("Volatility = ", self.v)
        print("Discrete time points =", self.N)
        print("Time-Step = ", self.dt)
        print("Underlyign domain = [", self.lb, ",", self.ub, "]")
        print("Discrete underlying points =", self.J)
        print("Underlying-Step = ", self.dS)
        print("---------------------------------------------")
