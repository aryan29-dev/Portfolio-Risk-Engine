import numpy as np
import pandas as pd

trading_days = 252

# simple arithmetic to convert prices to returns
def prices_to_returns(prices_matrix):
    return (prices_matrix[1:] / prices_matrix[:-1]) - 1

# annualized expected return calculation
def annualized_return(mean_daily_returns, weights):
    daily_portfolio_return = np.sum(mean_daily_returns * weights)
    return daily_portfolio_return * trading_days

# annualized volatility calculation
def annualized_volatility(cov_daily, weights):
    portfolio_variance = np.dot(weights, np.dot(cov_daily, weights))
    return np.sqrt(portfolio_variance * trading_days)

# sharpe ratio calculation
def sharpe_ratio(annual_ret, annual_vol, risk_free_rate):
    if annual_vol <= 0:
        return np.nan
    return (annual_ret - risk_free_rate) / annual_vol

# portfolio equity curve calculation
def portfolio_equity_curve(returns_matrix, weights, start_value=10000):
    daily_portfolio_returns = np.sum(returns_matrix * weights, axis=1)
    equity_curve = np.zeros(len(daily_portfolio_returns) + 1)
    equity_curve[0] = start_value
    for i in range(len(daily_portfolio_returns)):
        equity_curve[i + 1] = equity_curve[i] * (1 + daily_portfolio_returns[i])
    return equity_curve

# max drawdown calculation (MDD)
def max_drawdown(equity_curve):
    peak_value = equity_curve[0]
    worst_drawdown = 0.0
    for value in equity_curve:
        if value > peak_value:
            peak_value = value
        drawdown = (value / peak_value) - 1
        if drawdown < worst_drawdown:
            worst_drawdown = drawdown
    return worst_drawdown

# value at risk (VaR) and conditional value at risk (CVaR) calculation
def var_cvar(daily_portfolio_returns, confidence=0.95):
    sorted_returns = np.sort(daily_portfolio_returns)
    index = int((1 - confidence) * len(sorted_returns))
    var_return = sorted_returns[index]
    tail_losses = sorted_returns[:index + 1]
    cvar_return = np.mean(tail_losses)
    return -var_return, -cvar_return

# generate random weights that sum to 1
def random_weights(num_assets):
    weights = np.random.random(num_assets)
    weights = weights / np.sum(weights)
    return weights
