
"""
Created on Mon 26 June 2023

@author: effiehan
"""

# -*- coding: utf-8 -*-
import numpy as np

S = 100  # initial stock price
K = 105  # strike price
r = 0.02  # risk-free rate
T = 1.0  # time to expiration
sigma = 0.2  # volatility
num_simulations = int(3e5)  # number of simulations
num_intervals = 1000  # number of time intervals

dt = T / num_intervals  # time step
stock_price = np.zeros((num_intervals + 1, num_simulations))
stock_price[0] = S
np.random.seed(0)
standard_normal = np.random.standard_normal(size=(num_intervals, num_simulations))

for t in range(1, num_intervals + 1):
    stock_price[t] = (stock_price[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
                 + sigma * np.sqrt(dt) * standard_normal[t - 1]))


payoff = np.maximum(stock_price[-1] - K, 0)  # payoff of the call option
discounted_payoff = np.exp(-r * T) * payoff  # discount the payoff back to today
option_price = discounted_payoff.mean()  # calculate the option price
dS = 0.01  # small change in stock price
stock_price_bump = stock_price + dS
payoff_bump = np.maximum(stock_price_bump[-1] - K, 0)
discounted_payoff_bump = np.exp(-r * T) * payoff_bump
option_price_bump = discounted_payoff_bump.mean()
delta = (option_price_bump - option_price) / dS  # calculate delta

import matplotlib.pyplot as plt

deltas = []
initial_prices = np.arange(80, 120, 1)

for S in initial_prices:
    stock_price[0] = S
    for t in range(1, num_intervals + 1):
        stock_price[t] = (stock_price[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
                 + sigma * np.sqrt(dt) * standard_normal[t - 1]))

    payoff = np.maximum(stock_price[-1] - K, 0)
    discounted_payoff = np.exp(-r * T) * payoff
    option_price = discounted_payoff.mean()

    stock_price_bump = stock_price + dS
    print(stock_price)
    print(stock_price_bump)
    payoff_bump = np.maximum(stock_price_bump[-1] - K, 0)
    discounted_payoff_bump = np.exp(-r * T) * payoff_bump
    option_price_bump = discounted_payoff_bump.mean()

    delta = (option_price_bump - option_price) / dS
    deltas.append(delta)

plt.plot(initial_prices, deltas)
plt.xlabel('Initial Stock Price')
plt.ylabel('Delta')
plt.title('Delta of European Call Option')
plt.show()

