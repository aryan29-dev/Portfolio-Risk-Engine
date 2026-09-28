# Portfolio Risk Engine

Python-based portfolio analysis and optimization tool that uses Monte Carlo simulation to evaluate thousands of possible portfolio allocations using historical market data.

The engine simulates 5,000 long-only, fully invested portfolios and selects the allocation with the highest Sharpe ratio, representing the strongest risk-adjusted return among the portfolios sampled.

## Overview

The Portfolio Risk Engine:

- Retrieves historical market prices using `yfinance`.
- Converts historical prices into daily returns.
- Generates 5,000 randomized portfolio allocations.
- Calculates annualized return, volatility, and Sharpe ratio for each portfolio.
- Selects the portfolio with the highest Sharpe ratio.
- Evaluates downside risk using maximum drawdown, Value at Risk (VaR), and Conditional Value at Risk (CVaR).
- Visualizes portfolio performance through an equity curve and risk-return scatter plot.

Because portfolio weights are generated randomly, simulation results may vary slightly between runs.

## Features

- **Historical market data:** Downloads and prepares equity price data using `yfinance`.

- **Return calculations:** Converts historical prices into daily return series for portfolio analysis.

- **Monte Carlo simulation:** Generates 5,000 randomized long-only portfolios whose weights sum to 100%.

- **Portfolio optimization:** Identifies the simulated portfolio with the highest Sharpe ratio.

- **Annualized performance metrics:**
  - Expected return
  - Volatility
  - Sharpe ratio

- **Downside risk analysis:**
  - Maximum Drawdown (MDD)
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)

- **Visualizations:**
  - Optimal portfolio equity curve
  - Risk-return scatter plot of all simulated portfolios

## How It Works

### 1. Market Data

Historical daily prices are retrieved for a selected group of equities and converted into daily returns.

### 2. Monte Carlo Simulation

The engine generates 5,000 randomized portfolio weight combinations.

Each simulated portfolio is:

- Fully invested
- Long-only
- Evaluated using the same historical return data

For every allocation, the engine calculates expected annualized return, volatility, and Sharpe ratio.

### 3. Portfolio Selection

The portfolio with the highest Sharpe ratio is selected as the optimal allocation from the simulated set.

This represents the portfolio with the strongest risk-adjusted performance among the allocations evaluated during the simulation.

### 4. Risk Evaluation

The selected portfolio is then evaluated using additional downside-risk measures:

- **Maximum Drawdown:** Largest peak-to-trough decline in portfolio value.
- **Value at Risk (VaR):** Estimates a loss threshold for the portfolio at a specified confidence level.
- **Conditional Value at Risk (CVaR):** Estimates the average loss when returns fall beyond the VaR threshold.

An equity curve is also generated to show how the selected portfolio would have performed over the historical period.

## Project Structure

```text
portfolio-risk-engine/
├── src/
│   ├── main.py            # Runs Monte Carlo simulation and analysis
│   └── risk_engine.py     # Core portfolio and risk calculations
├── data/
│   └── prices.csv         # Historical market price data
├── make_prices_csv.py     # Downloads and prepares price data
├── requirements.txt
├── LICENSE
└── README.md
```

## Tech Stack

| Area | Technology |
| --- | --- |
| Language | Python |
| Market Data | yfinance |
| Data Processing | pandas |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Portfolio Optimization | Monte Carlo simulation |
| Risk Analysis | Sharpe Ratio, MDD, VaR, CVaR |

## Libraries

- `NumPy`
- `pandas`
- `yfinance`
- `Matplotlib`