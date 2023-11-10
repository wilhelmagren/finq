"""
Script to download the last 3 months of historical OHLC data from the 
OMXS30 index on the OMX market.

Optimizes portfolio weights by minimizing the mean-variance expression.
Allows the user to set a specified risk tolerance, where a larger
number is less tolerance for risk, and vice versa.

Lastly plots the expected returns and volatility of the optimized
portfolio compared to a number of randomly sampled portfolios.

File created: 2023-11-10
Last updated: 2023-11-10
"""

from finq import Portfolio
from finq.datasets import OMXS30
from finq.formulas import mean_variance

dataset = OMXS30(save=True)
dataset = dataset.run("2y")

portfolio = Portfolio(dataset)
portfolio.initialize_random_weights(
    "lognormal",
    size=(len(dataset), 1),
)

risk_tolerance = 1

portfolio.set_objective_function(
    mean_variance,
    risk_tolerance * portfolio.daily_covariance(),
    portfolio.daily_returns_mean(),
)

portfolio.set_objective_bounds(
    [(0, 0.2) for _ in range(len(dataset))],
)

portfolio.optimize(
    method="COBYLA",
    options={"maxiter": 1000},
)

portfolio.plot_mean_variance(n_samples=10000, figsize=(8, 5))

