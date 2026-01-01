import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from risk_engine import (
    prices_to_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    portfolio_equity_curve,
    max_drawdown,
    var_cvar,
    random_weights
)

risk_free_rate = 0.04
num_portfolios = 5000

def main():
    df = pd.read_csv("data/prices.csv", parse_dates=["Date"])
    tickers = list(df.columns[1:])
    prices_matrix = df[tickers].to_numpy(dtype=float)
    returns_matrix = prices_to_returns(prices_matrix)

    mean_daily_returns = np.mean(returns_matrix, axis=0)
    cov_daily = np.cov(returns_matrix, rowvar=False)

    best_sharpe = -999999
    best_result = None

    all_results = []

    # Monte Carlo simulation
    for i in range(num_portfolios):
        weights = random_weights(len(tickers))

        annual_returns = annualized_return(mean_daily_returns, weights)
        annual_volatility = annualized_volatility(cov_daily, weights)
        sharpe = sharpe_ratio(annual_returns, annual_volatility, risk_free_rate)

        equity_curve = portfolio_equity_curve(returns_matrix, weights)
        mdd = max_drawdown(equity_curve)

        daily_port_returns = np.sum(returns_matrix * weights, axis=1)

        var95, cvar95 = var_cvar(daily_port_returns, confidence=0.95)

        all_results.append([annual_returns, annual_volatility, sharpe])

        # best Sharpe portfolio as the for-loop runs
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_result = [
                annual_returns,
                annual_volatility,
                sharpe,
                mdd,
                var95,
                cvar95,
                weights
            ]

    best_ann_ret, best_ann_vol, best_sr, best_mdd, best_var95, best_cvar95, best_weights = best_result

    print("\n===== BEST PORTFOLIO GENERATED FROM RANDOMNESS (MAX SHARPE) =====\n")

    print("Optimal Weights:")
    for ticker, weight in zip(tickers, best_weights):
        print(f"{ticker}: {weight:.4f}")

    print("\nKey Metrics:")
    print(f"Annual Return: {best_ann_ret:.4f}")
    print(f"Annual Volatility: {best_ann_vol:.4f}")
    print(f"Sharpe Ratio: {best_sr:.4f}")
    print(f"Max Drawdown: {best_mdd:.4f}")
    print(f"VaR 95% (daily): {best_var95:.4f}")
    print(f"CVaR 95% (daily): {best_cvar95:.4f}")

    best_equities = portfolio_equity_curve(returns_matrix, best_weights)
    
    equity_dates = df["Date"].iloc[1:]
    equity_dates = equity_dates.reset_index(drop=True)

    equity_dates = pd.concat([pd.Series([df["Date"].iloc[0]]), equity_dates], ignore_index=True)

    plt.plot(equity_dates, best_equities)
    plt.title("Best Portfolio Equity Curve")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    
    plt.gcf().autofmt_xdate()
    plt.show()

    volatilities = [row[1] for row in all_results]
    returns = [row[0] for row in all_results]

    plt.scatter(volatilities, returns, s=5)
    plt.title("Random Portfolios")
    plt.xlabel("Annual Volatility")
    plt.ylabel("Annual Return")

    plt.gcf().autofmt_xdate()
    plt.show()

if __name__ == "__main__":
    main()
