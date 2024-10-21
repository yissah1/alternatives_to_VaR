import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme
from scipy.stats import genpareto
import os

def get_tickers():
    """
    Define and return all ticker symbols categorized into different sectors.

    Returns:
        list: A combined list of all ticker symbols.
    """
    ticker_banks = ['JPM', 'BAC', 'HSBA.L', 'CBA', 'RY']
    ticker_other_fin = ['SCHW', 'V', 'MA', 'BRK-B', 'BLK']
    ticker_tech = ['AAPL', 'AMZN', '^GSPC', 'MSFT', 'NVDA']
    ticker_others = ['JNJ', 'EUR=X', 'GLD', 'BZ=F', 'SPY']

    tickers = ticker_banks + ticker_other_fin + ticker_tech + ticker_others
    return tickers

def get_weights(tickers):
    """
    Define and return portfolio weights for each ticker.

    Args:
        tickers (list): List of ticker symbols.

    Returns:
        dict: A dictionary mapping each ticker to its portfolio weight.
    """
    weights = {
        'JPM': 0.05, 'BAC': 0.05, 'HSBA.L': 0.05, 'CBA': 0.05, 'RY': 0.05,      # Banks
        'SCHW': 0.05, 'V': 0.05, 'MA': 0.05, 'BRK-B': 0.05, 'BLK': 0.05,      # Other Financials
        'AAPL': 0.05, 'AMZN': 0.05, '^GSPC': 0.05, 'MSFT': 0.05, 'NVDA': 0.05, # Tech
        'JNJ': 0.05, 'EUR=X': 0.05, 'GLD': 0.05, 'BZ=F': 0.05, 'SPY': 0.05    # Others
    }

    # Verify that all tickers have weights
    missing_weights = set(tickers) - set(weights.keys())
    if missing_weights:
        raise ValueError(f"Missing weights for tickers: {missing_weights}")

    # Ensure that the weights sum to 1
    total_weight = sum(weights[ticker] for ticker in tickers)
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"Total portfolio weight must sum to 1. Currently, it sums to {total_weight}")

    return weights

def download_asset_data(tickers, start_date, end_date):
    """
    Download adjusted closing price data for the specified tickers and date range.

    Args:
        tickers (list): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing adjusted closing prices.
    """
    print(f"Downloading data for tickers: {tickers}")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Ensure the data is a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Forward fill missing data to handle non-trading days or missing entries
    data.ffill(inplace=True)

    print("Data download completed.\n")
    return data

def calculate_daily_returns(price_data):
    """
    Calculate daily percentage returns for each asset.

    Args:
        price_data (pd.DataFrame): DataFrame of adjusted closing prices.

    Returns:
        pd.DataFrame: DataFrame of daily returns.
    """
    daily_returns = price_data.pct_change().dropna()
    print("Daily returns calculated.\n")
    return daily_returns

def calculate_portfolio_returns(daily_returns, weights):
    """
    Calculate portfolio daily returns based on individual asset returns and weights.

    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        weights (dict): Dictionary of asset weights.

    Returns:
        pd.Series: Series of portfolio daily returns.
    """
    # Align weights with the order of columns in daily_returns
    weights_array = np.array([weights[ticker] for ticker in daily_returns.columns])

    # Compute portfolio daily returns
    portfolio_daily_returns = daily_returns.dot(weights_array)
    portfolio_daily_returns.name = 'Portfolio_Returns'
    print("Portfolio daily returns calculated.\n")
    return portfolio_daily_returns

def calculate_rolling_returns(portfolio_returns, window=10, compounded=False):
    """
    Calculate rolling portfolio returns over a specified window.

    Args:
        portfolio_returns (pd.Series): Series of portfolio daily returns.
        window (int, optional): Rolling window size in days. Defaults to 10.
        compounded (bool, optional): Whether to calculate compounded returns. Defaults to False.

    Returns:
        pd.Series: Series of rolling portfolio returns.
    """
    if compounded:
        # Compounded returns: (1 + r1) * (1 + r2) * ... * (1 + rN) - 1
        rolling_returns = (1 + portfolio_returns).rolling(window=window).apply(np.prod, raw=True) - 1
    else:
        # Simple sum of daily returns over the window
        rolling_returns = portfolio_returns.rolling(window=window).sum()

    print(f"Ten-day rolling portfolio returns calculated.\n")
    return rolling_returns

def value_at_risk(returns, confidence_level=0.95):
    """
    Compute the Value-at-Risk (VaR) metric at a given confidence level.

    Args:
        returns (pd.Series): Series of portfolio returns.
        confidence_level (float): Confidence level for VaR (e.g., 0.95 for 95%).

    Returns:
        float: VaR value.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    # Since VaR is a loss, we consider the lower tail
    var = returns.quantile(1 - confidence_level, interpolation="higher")
    return var

def expected_shortfall(returns, confidence_level=0.95):
    """
    Compute the Conditional Value-at-Risk (CVaR) at a given confidence level.

    Args:
        returns (pd.Series): Series of portfolio returns.
        confidence_level (float): Confidence level for CVaR (e.g., 0.95 for 95%).

    Returns:
        float: CVaR value.
    """
    var = value_at_risk(returns, confidence_level)
    # Select returns that are worse than the VaR
    worst_returns = returns[returns <= var]
    # CVaR is the mean of these worst returns
    cvar = worst_returns.mean()
    return cvar

def evt_value_at_risk(returns, confidence_level=0.98, threshold_quantile=0.98):
    """
    Compute the Value-at-Risk using Extreme Value Theory (EVT) and Generalized Pareto Distribution (GPD).

    Args:
        returns (pd.Series): Series of portfolio returns.
        confidence_level (float): Confidence level for EVT-Based VaR (e.g., 0.95 for 95% VaR).
        threshold_quantile (float): Quantile threshold for extreme events (e.g., 0.95 for top 5% extreme losses).

    Returns:
        float: EVT-Based VaR value.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    # Sort returns to find the threshold
    sorted_returns = np.sort(returns)

    # Define threshold as the return at the threshold quantile (e.g., top 5% extreme losses)
    threshold = sorted_returns[int((1 - threshold_quantile) * len(sorted_returns))]

    # Extract returns that exceed the threshold (extreme losses)
    excess_returns = returns[returns < threshold] - threshold  # Losses beyond the threshold

    # Fit the Generalized Pareto Distribution (GPD) to the excess returns
    params = genpareto.fit(-excess_returns)  # Fit GPD to negative excess returns
    shape, loc, scale = params

    # Compute the VaR at the specified confidence level using the GPD
    tail_prob = (1 - confidence_level) / (1 - threshold_quantile)
    evt_var = threshold - genpareto.ppf(tail_prob, shape, loc=loc, scale=scale)

    return evt_var

def plot_returns(portfolio_daily_returns, portfolio_rolling_returns):
    """
    Plot daily and rolling portfolio returns.

    Args:
        portfolio_daily_returns (pd.Series): Series of portfolio daily returns.
        portfolio_rolling_returns (pd.Series): Series of rolling portfolio returns.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_daily_returns.index, portfolio_daily_returns, label='Daily Portfolio Returns', alpha=0.5)
    plt.plot(portfolio_rolling_returns.index, portfolio_rolling_returns, label='10-Day Rolling Portfolio Returns', linewidth=2)
    plt.title('Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_risk_metrics(returns, var, cvar, evt_var, confidence_level=0.95):
    """
    Plot the distribution of returns with VaR, CVaR, and EVT-Based VaR thresholds.

    Args:
        returns (pd.Series): Series of portfolio returns.
        var (float): Value-at-Risk.
        cvar (float): Conditional Value-at-Risk.
        evt_var (float): EVT-Based Value-at-Risk.
        confidence_level (float): Confidence level used for VaR and CVaR.
    """
    plt.figure(figsize=(10,6))
    returns.hist(bins=50, alpha=0.5, label='Returns', color='skyblue')
    plt.axvline(var, color='r', linestyle='--', label=f'VaR ({int(confidence_level*100)}%)')
    plt.axvline(cvar, color='b', linestyle='--', label=f'CVaR ({int(confidence_level*100)}%)')
    plt.axvline(evt_var, color='g', linestyle='--', label=f'EVT VaR ({int(confidence_level*100)}%)')
    plt.title('Portfolio Return Distribution with Risk Metrics')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def process_portfolio_risk_metrics(portfolio_returns, training_period=250,
                                   confidence_levels={'VaR':0.95, 'ES':0.95, 'EVT_VaR':0.95},
                                   composite_weights={'VaR':0.4, 'ES':0.3, 'EVT_VaR':0.3},
                                   output_dir='risk_metrics'):
    """
    Process risk metrics for the portfolio returns over a rolling window, including a composite risk metric.

    Args:
        portfolio_returns (pd.Series): Series of portfolio daily returns.
        training_period (int, optional): Number of days to use for training. Defaults to 250.
        confidence_levels (dict, optional): Dictionary with keys 'VaR', 'ES', 'EVT_VaR' and their confidence levels.
        composite_weights (dict, optional): Weights for composite risk metric. Defaults to {'VaR':0.4, 'ES':0.3, 'EVT_VaR':0.3}.
        output_dir (str, optional): Directory to save the CSV results. Defaults to 'risk_metrics'.

    Returns:
        pd.DataFrame: DataFrame containing risk metrics, composite risk, and exceedance flags.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a list to collect results
    results = []

    # Define the columns for the result DataFrame
    columns = [
        'date', 'VaR_99%', 'CVaR_97.5%', 'EVT_VaR_99%', 'Composite_Risk_Metric',
        'testing_return', 'exceeds_VaR_99%', 'exceeds_CVaR_97.5%', 'exceeds_EVT_VaR_99%',
        'exceeds_Composite_Risk'
    ]

    # Total number of observations
    total_obs = len(portfolio_returns)

    print(f"Processing risk metrics over a training period of {training_period} days...\n")

    # Iterate over the testing period
    for i in range(training_period, total_obs):
        # Define the training data up to the current day
        training_data = portfolio_returns.iloc[i - training_period:i]

        # Define the testing data point
        testing_return = round(portfolio_returns.iloc[i], 4)
        testing_date = portfolio_returns.index[i]

        # Calculate VaR at 99% confidence level
        var = round(value_at_risk(training_data, confidence_level=confidence_levels['VaR']), 4)

        # Calculate CVaR at 97.5% confidence level
        cvar = round(expected_shortfall(training_data, confidence_level=confidence_levels['ES']), 4)

        # Compute EVT-Based VaR at 99% confidence level
        evt_var = round(evt_value_at_risk(training_data, confidence_level=confidence_levels['EVT_VaR']), 4)

        # Calculate Composite Risk Metric
        composite_risk = (
            composite_weights['VaR'] * var +
            composite_weights['ES'] * cvar +
            composite_weights['EVT_VaR'] * evt_var
        )

        # Determine if testing return exceeds the risk metrics
        exceeds_var = testing_return < var  # Since VaR is a loss threshold
        exceeds_cvar = testing_return < cvar
        exceeds_evt_var = testing_return < evt_var
        exceeds_composite = testing_return < composite_risk

        # Compile the result into a dictionary
        result = {
            'date': testing_date,
            'VaR_99%': var,
            'CVaR_97.5%': cvar,
            'EVT_VaR_99%': evt_var,
            'Composite_Risk_Metric': composite_risk,
            'testing_return': testing_return,
            'exceeds_VaR_99%': exceeds_var,
            'exceeds_CVaR_97.5%': exceeds_cvar,
            'exceeds_EVT_VaR_99%': exceeds_evt_var,
            'exceeds_Composite_Risk': exceeds_composite
        }

        # Append the result to the list
        results.append(result)

        # Optional: Print progress every 500 iterations
        if (i - training_period) % 500 == 0 and (i - training_period) != 0:
            print(f'Processed {i - training_period} out of {total_obs - training_period} observations.')

    # Convert the list of results to a DataFrame
    df_result = pd.DataFrame(results, columns=columns)

    # Save the result as a CSV file
    csv_filename = os.path.join(output_dir, 'portfolio_risk_metrics.csv')
    df_result.to_csv(csv_filename, index=False)
    print(f'\nRisk metrics processing completed. Results saved to {csv_filename}')

    return df_result

def backtest_risk_metrics(risk_metrics_df, confidence_levels):
    """
    Backtest the risk metrics by comparing exceedance frequencies with expected frequencies.

    Args:
        risk_metrics_df (pd.DataFrame): DataFrame containing risk metrics and exceedance flags.
        confidence_levels (dict): Dictionary with keys 'VaR', 'ES', 'EVT_VaR' and their confidence levels.

    Returns:
        None
    """
    total_tests = len(risk_metrics_df)

    # Calculate actual exceedance rates
    actual_var_exceed = risk_metrics_df['exceeds_VaR_99%'].mean()
    actual_cvar_exceed = risk_metrics_df['exceeds_CVaR_97.5%'].mean()
    actual_evt_var_exceed = risk_metrics_df['exceeds_EVT_VaR_99%'].mean()
    actual_composite_exceed = risk_metrics_df['exceeds_Composite_Risk'].mean()

    # Expected exceedance rates
    expected_var_exceed = 1 - confidence_levels['VaR']
    expected_cvar_exceed = 1 - confidence_levels['ES']
    expected_evt_var_exceed = 1 - confidence_levels['EVT_VaR']
    # For composite, assume it's a weighted average of expected exceedances
    expected_composite_exceed = (
        (1 - confidence_levels['VaR']) * 0.4 +
        (1 - confidence_levels['ES']) * 0.3 +
        (1 - confidence_levels['EVT_VaR']) * 0.3
    )

    print(f"\nBacktesting Risk Metrics:")
    print(f"VaR 99% - Expected Exceedance Rate: {expected_var_exceed:.2%}, Actual: {actual_var_exceed:.2%}")
    print(f"CVaR 97.5% - Expected Exceedance Rate: {expected_cvar_exceed:.2%}, Actual: {actual_cvar_exceed:.2%}")
    print(f"EVT VaR 99% - Expected Exceedance Rate: {expected_evt_var_exceed:.2%}, Actual: {actual_evt_var_exceed:.2%}")
    print(f"Composite Risk Metric - Expected Exceedance Rate: {expected_composite_exceed:.2%}, Actual: {actual_composite_exceed:.2%}")

def plot_composite_risk_metrics(risk_metrics_df):
    """
    Plot the Composite Risk Metric over time.

    Args:
        risk_metrics_df (pd.DataFrame): DataFrame containing risk metrics and exceedance flags.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(risk_metrics_df['date'], risk_metrics_df['Composite_Risk_Metric'], label='Composite Risk Metric', color='purple')
    plt.title('Composite Risk Metric Over Time')
    plt.xlabel('Date')
    plt.ylabel('Composite Risk')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_exceedances(risk_metrics_df, confidence_level=0.99):
    """
    Plot portfolio returns and highlight exceedance events based on VaR, CVaR, EVT VaR, and Composite Risk.

    Args:
        risk_metrics_df (pd.DataFrame): DataFrame containing risk metrics and exceedance flags.
        confidence_level (float): Confidence level used for VaR and CVaR.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(risk_metrics_df['date'], risk_metrics_df['testing_return'], label='Testing Return', color='blue')

    # Plot VaR, CVaR, EVT VaR, and Composite Risk as horizontal lines (average values)
    avg_var = risk_metrics_df['VaR_99%'].mean()
    avg_cvar = risk_metrics_df['CVaR_97.5%'].mean()
    avg_evt_var = risk_metrics_df['EVT_VaR_99%'].mean()
    avg_composite = risk_metrics_df['Composite_Risk_Metric'].mean()

    plt.axhline(y=avg_var, color='r', linestyle='--', label='VaR 99% (Avg)')
    plt.axhline(y=avg_cvar, color='b', linestyle='--', label='CVaR 97.5% (Avg)')
    plt.axhline(y=avg_evt_var, color='g', linestyle='--', label='EVT VaR 99% (Avg)')
    plt.axhline(y=avg_composite, color='m', linestyle='--', label='Composite Risk Metric (Avg)')

    # Highlight exceedances
    exceed_var = risk_metrics_df[risk_metrics_df['exceeds_VaR_99%']]
    exceed_cvar = risk_metrics_df[risk_metrics_df['exceeds_CVaR_97.5%']]
    exceed_evt_var = risk_metrics_df[risk_metrics_df['exceeds_EVT_VaR_99%']]
    exceed_composite = risk_metrics_df[risk_metrics_df['exceeds_Composite_Risk']]

    plt.scatter(exceed_var['date'], exceed_var['testing_return'], color='red', label='Exceeds VaR 99%', marker='o')
    plt.scatter(exceed_cvar['date'], exceed_cvar['testing_return'], color='blue', label='Exceeds CVaR 97.5%', marker='x')
    plt.scatter(exceed_evt_var['date'], exceed_evt_var['testing_return'], color='green', label='Exceeds EVT VaR 99%', marker='^')
    plt.scatter(exceed_composite['date'], exceed_composite['testing_return'], color='magenta', label='Exceeds Composite Risk', marker='s')

    plt.title('Portfolio Testing Returns with Risk Metrics Exceedances')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_composite_risk_exceedances(risk_metrics_df):
    """
    Plot portfolio returns and highlight exceedance events based on the Composite Risk Metric.

    Args:
        risk_metrics_df (pd.DataFrame): DataFrame containing risk metrics and exceedance flags.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(risk_metrics_df['date'], risk_metrics_df['testing_return'], label='Testing Return', color='blue')
    plt.plot(risk_metrics_df['date'], risk_metrics_df['Composite_Risk_Metric'], label='Composite Risk Metric', color='purple')

    # Highlight exceedances
    exceed_composite = risk_metrics_df[risk_metrics_df['exceeds_Composite_Risk']]
    plt.scatter(exceed_composite['date'], exceed_composite['testing_return'], color='magenta', label='Exceeds Composite Risk', marker='s')

    plt.title('Portfolio Testing Returns with Composite Risk Metric Exceedances')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to execute the portfolio return computations and risk metrics.
    """
    # Step 1: Define tickers and weights
    tickers = get_tickers()
    weights = get_weights(tickers)

    # Step 2: Define the date range for data download
    date_start = '2007-01-01'
    date_end = '2022-01-01'

    # Step 3: Download asset data
    price_data = download_asset_data(tickers, date_start, date_end)

    # Step 4: Calculate daily returns
    daily_returns = calculate_daily_returns(price_data)

    # Step 5: Calculate portfolio daily returns
    portfolio_daily_returns = calculate_portfolio_returns(daily_returns, weights)

    # Step 6: Calculate ten-day rolling portfolio returns
    portfolio_10day_returns = calculate_rolling_returns(portfolio_daily_returns, window=10, compounded=False)

    # Step 7: Compute Risk Metrics for the Entire Portfolio
    print("Processing risk metrics for the portfolio...\n")
    training_period = 250  # Approximately 1 trading year
    confidence_levels = {
        'VaR': 0.99,       # 99% confidence for VaR
        'ES': 0.975,       # 97.5% confidence for CVaR
        'EVT_VaR': 0.99    # 99% confidence for EVT-Based VaR
    }
    composite_weights = {'VaR':0.4, 'ES':0.3, 'EVT_VaR':0.3}  # Example weights for composite risk metric
    risk_metrics_df = process_portfolio_risk_metrics(
        portfolio_returns=portfolio_daily_returns,
        training_period=training_period,
        confidence_levels=confidence_levels,
        composite_weights=composite_weights,
        output_dir='risk_metrics'
    )

    # Step 8: Plot the returns
    print("Plotting portfolio returns...")
    plot_returns(portfolio_daily_returns, portfolio_10day_returns)

    # Step 9: Plot Composite Risk Metric Over Time
    print("Plotting Composite Risk Metric Over Time...")
    plot_composite_risk_metrics(risk_metrics_df)

    # Step 10: Plot Risk Metrics Distribution
    print("Plotting Risk Metrics Distribution...")
    # Calculate overall VaR, CVaR, EVT VaR for plotting
    overall_var = value_at_risk(portfolio_daily_returns, confidence_level=confidence_levels['VaR'])
    overall_cvar = expected_shortfall(portfolio_daily_returns, confidence_level=confidence_levels['ES'])
    overall_evt_var = evt_value_at_risk(portfolio_daily_returns, confidence_level=confidence_levels['EVT_VaR'])

    plot_risk_metrics(portfolio_daily_returns, overall_var, overall_cvar, overall_evt_var, confidence_level=confidence_levels['VaR'])

    # Step 11: Plot Exceedances
    print("Plotting Exceedances of Risk Metrics...")
    plot_exceedances(risk_metrics_df, confidence_level=confidence_levels['VaR'])

    # Step 12: Plot Composite Risk Metric Exceedances
    print("Plotting Exceedances of Composite Risk Metric...")
    plot_composite_risk_exceedances(risk_metrics_df)

    # Step 13: Backtest Risk Metrics
    print("\nBacktesting Risk Metrics...")
    backtest_risk_metrics(risk_metrics_df, confidence_levels)

    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()
