# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon 26 June 2023

@author: effiehan
"""
# greeks are the relative change at time t
# DELTA :stock price grow from S0 to the current plot range [50-150],
# we want to know how the delta at time t if stock price is [50-100] at time t,
# then we should change st from [50-100] S0 stays the same

import numpy as np
from time import time
import pandas as pd
import matplotlib.pyplot as plt


class snowball_mc_vanilla_greek:
    def __init__(self, S,K, KI_Barrier, KO_Barrier, KO_Coupon, r, T, v, lockmonth, epsilon,simulations=3e5):
        self.S = S
        self.KI_Barrier = KI_Barrier
        self.KO_Barrier = KO_Barrier
        self.KO_Coupon = KO_Coupon
        self.r = r
        self.T = T
        self.v = v
        self.K=K
        self.lockmonth = lockmonth
        self.simulations = int(simulations)
        self.N = 252 * self.T  # number of days for the simulation
        self.dt = self.T / self.N  # delta t
        self.price_path = []  # prices of all simulation paths
        self.option_price = np.nan
        self.delta = np.nan
        self.gamma = np.nan
        self.vega = np.nan
        self.epsilon=epsilon

    def compute_price_(self):
        self.pv_path = []
        self.stock_prices = None
        # Vectorize simulation: generate all random numbers at once
        e = np.random.normal(0, 1, (self.simulations, self.N))

        self.stock_prices = np.cumprod(np.exp((self.r - 0.5 * self.v ** 2) * self.dt \
                                              + self.v * np.sqrt(self.dt) * e), axis=1) *self.S
        n = int(1.0 / self.dt / 12.0)
        s = slice(n - 1, self.N, n)
        stock_prices_slice = self.stock_prices[:, s]

        if self.lockmonth != 0:
            stock_prices_slice = stock_prices_slice[:, self.lockmonth - 1:]

        KO_group_indicator = (stock_prices_slice.max(axis=1) >= self.KO_Barrier)
        KO_group = stock_prices_slice[KO_group_indicator]
        idx = np.argmax(KO_group >= self.KO_Barrier, axis=1)
        idx = idx if self.lockmonth == 0 else idx + (self.lockmonth - 1)
        time_to_KO = (idx + 1) / 12.0
        pv_KO = (self.KO_Coupon * time_to_KO) * np.exp(-self.r * time_to_KO)

        KI_group = (self.stock_prices[~KO_group_indicator])
        indicator_KI = KI_group.min(axis=1) < self.KI_Barrier
        indicator_FP = KI_group[:, -1] < self.K
        pv_no_KO = (1 - indicator_KI) * (self.KO_Coupon * np.exp(-self.r *  self.T)) \
                    + indicator_KI * indicator_FP * (KI_group[:, -1] - self.K) * np.exp(-self.r * self.T) \
                   + indicator_KI * (1 - indicator_FP) * 0
        self.pv_path = np.concatenate([pv_KO, pv_no_KO])
        self.option_price= np.mean(self.pv_path)
        print(KI_group.shape,KO_group.shape)
    def conpute_delta_(self,S0):
        self.S=S0
        self.compute_price_()
        C1=self.option_price
        self.S=S0+self.epsilon
        self.compute_price_()
        C2=self.option_price
        delta=(C2-C1)/self.epsilon
        print(self.S,C1,C2,delta)
        return delta

    def compute_greeks_(self, S0):
        """"compute greeks of snowball option"""
        epsilon = 0.1
        # S = S0 * (1 - epsilon)
        self.S = S0 * (1 - epsilon)
        self.compute_price_()
        P1 = self.option_price
        # S = S0 * (1 + epsilon)
        self.S = S0 * (1 + epsilon)
        self.compute_price_()
        P2 = self.option_price

        # price with S = S0 and vol = vol + epsilon for vega
        self.S = S0
        self.v = self.v + epsilon
        self.compute_price_()
        P3 = self.option_price

        # back to original and price option price
        self.v = self.v - epsilon
        self.compute_price_()
        P0 = self.option_price

        self.delta = (P2 - P1) / (2 * S0 * epsilon)
        self.gamma = (P1 + P2 - 2 * P0) / (S0 ** 2 * epsilon ** 2)
        self.vega = (P3 - P0) / epsilon
        return self.delta, self.gamma, self.vega
        # return self.delta

    def plot_greek_(self, lower_bound, upper_bound):
        prices = np.arange(lower_bound, upper_bound, 0.01).tolist()
        # results = [self.conpute_delta_(price) for price in prices]
        results = [self.compute_greeks_(price) for price in prices]
        # df = pd.DataFrame(results, columns=['Delta'], index=prices)
        df = pd.DataFrame(results, columns=['Delta', 'Gamma', 'Vega'], index=prices)
        df['Delta'].plot(title='Delta values for different prices')
        # df['Gamma'].plot(title='Gamma values for different prices')
        # df['Vega'].plot(title='Vega values for different prices')
        plt.show()
        return df

if __name__ == '__main__':
    S = 1
    KI_Barrier = 0.85
    KO_Barrier = 1.03
    KO_Coupon = 0.2
    r = 0.03
    T = 1
    K=1
    v = 0.13
    lockmonth = 0
    epsilon=0.01
    start = time()
    mc = snowball_mc_vanilla_greek(S,K, KI_Barrier, KO_Barrier, KO_Coupon, r, T, v, lockmonth,epsilon)
    df=mc.plot_greek_(0.5,1.5)

    print('it took ',time() - start,'seconds')


