# -*- coding: utf-8 -*-
"""
Created on Mon 26 June 2023

@author: effiehan
"""

import numpy as np
from time import time
import pandas as pd
import matplotlib.pyplot as plt


class snowball_mc_vanilla_vector:
    def __init__(self, S, KI_Barrier, KO_Barrier, KO_Coupon, r, T, v, lockmonth, simulations=3e5):
        self.S = S
        self.KI_Barrier = KI_Barrier
        self.KO_Barrier = KO_Barrier
        self.KO_Coupon = KO_Coupon
        self.r = r
        self.T = T
        self.v = v
        self.lockmonth = lockmonth
        self.simulations = int(simulations)
        self.N = 252 * self.T  # number of days for the simulation
        self.dt = self.T / self.N  # delta t
        self.price_path = []  # prices of all simulation paths
        self.option_price = np.nan
        self.delta = np.nan
        self.gamma = np.nan
        self.vega = np.nan

    def compute_price_(self):
        # reset 雪球payoff pv
        self.pv_path = []
        self.stock_prices = None
        # Vectorize simulation: generate all random numbers at once （向量化比loop 更快）
        e = np.random.normal(0, 1, (self.simulations, self.N))
        #标的价格 cumprod vs cumsum axis =1 for each column st+1=st*exp()
        self.stock_prices = np.cumprod(np.exp((self.r - 0.5 * self.v ** 2) * self.dt \
                                              + self.v * np.sqrt(self.dt) * e), axis=1) * self.S
        #double check 每个月天数
        n = int(1.0 / self.dt / 12.0)
        s = slice(n - 1, self.N, n) #将stock_prices 从第20个position到N 每隔21个取一个值
        stock_prices_slice = self.stock_prices[:, s]  #每条路径都取到了12个观察日

        if self.lockmonth != 0: # 取过了锁定期的观察日价格
            stock_prices_slice = stock_prices_slice[:, self.lockmonth - 1:]
        # 选择有出现过敲出的路径
        KO_group_indicator = (stock_prices_slice.max(axis=1) >= self.KO_Barrier)
        KO_group = stock_prices_slice[KO_group_indicator]
        #argmax 得到第一个大于KO_barrier 的标的position
        idx = np.argmax(KO_group >= self.KO_Barrier, axis=1)
        #通过position 反推是第几个月
        idx = idx if self.lockmonth == 0 else idx + (self.lockmonth - 1)
        time_to_KO = (idx + 1)*21 / 365.0
        #计算pv
        pv_KO = (self.KO_Coupon * time_to_KO) * np.exp(-self.r * time_to_KO)

        KI_group = (self.stock_prices[~KO_group_indicator])
        #在未敲出组 用indicator 区分 未敲入未敲出 or 未敲入敲出
        indicator_KI = KI_group.min(axis=1) < self.KI_Barrier
        indicator_FP = KI_group[:, -1] < self.S # 这个时候put 生效
        pv_no_KO = (1 - indicator_KI) * (self.KO_Coupon * np.exp(-self.r *  self.T)) \
                    + indicator_KI * indicator_FP * (KI_group[:, -1] - self.S) * np.exp(-self.r * self.T) \
                   + indicator_KI * (1 - indicator_FP) * 0
        self.pv_path = np.concatenate([pv_KO, pv_no_KO])
        self.option_price = np.mean(self.pv_path)

    def bisection_method_(self, lower_bound, upper_bound, tolerance=1e-6):
        mid_point = (lower_bound + upper_bound) / 2.0
        while (upper_bound - lower_bound) / 2.0 > tolerance:
            self.KO_Coupon = mid_point
            self.compute_price_()
            if abs(self.option_price) < tolerance:
                return mid_point
            elif self.option_price > 0:
                upper_bound = mid_point
            else:
                lower_bound = mid_point
            mid_point = (lower_bound + upper_bound) / 2.0
        return mid_point

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
        # self.gamma = (P1 + P2 - 2 * P0) / (S0 ** 2 * epsilon ** 2)
        # self.vega = (P3 - P0) / epsilon
        # return self.delta, self.gamma, self.vega
        return self.delta

    def exp_fin_diff_delta(self,S0):

        self.S=S0
        self.compute_price_()
        C1=self.option_price
        eplsion = S0 * 0.01
        self.S = S0 + eplsion
        self.compute_price_()
        C2=self.option_price
        delta = (C2 - C1) / eplsion
        return delta

    def plot_greek_(self, lower_bound, upper_bound):
        prices = np.arange(lower_bound, upper_bound, 0.1).tolist()
        results = [self.exp_fin_diff_delta(price) for price in prices]
        # results = [self.compute_greeks_(price) for price in prices]
        df = pd.DataFrame(results, columns=['Delta'], index=prices)
        # df = pd.DataFrame(results, columns=['Delta', 'Gamma', 'Vega'], index=prices)
        df.plot(title='Greek values for different prices')
        plt.show()


if __name__ == '__main__':
    S = 1
    KI_Barrier = 0.85
    KO_Barrier = 1.03
    KO_Coupon = 0.2
    r = 0.03
    T = 1
    v = 0.13
    lockmonth = 0
    start = time()
    mc = snowball_mc_vanilla_vector(S, KI_Barrier, KO_Barrier, KO_Coupon, r, T, v, lockmonth)
    mc.compute_price_()
    print(mc.option_price)
    print(time() - start)

    start = time()
    print('start works')
    # coupon = mc.bisection_method_(0, 0.1)
    mc.plot_greek_(0.5, 1.5)
    print(time() - start)

