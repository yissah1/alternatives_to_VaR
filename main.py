# Define the ticker for the assets
ticker_banks = ['JPM','BAC','HSBA.L','CBA','RY']
ticker_other_fin = ['SCHW', 'V', 'MA', 'BRK-B', 'BLK']
ticket_tech = ['AAPL', 'AMZN', '^GSPC', 'MSFT','NVDA']
ticker_others = ['JNJ', 'EUR=X', 'GLD', 'BZ=F','SPY']

tickers = ticker_banks + ticker_other_fin + ticket_tech + ticker_others

#tickers = ['JPM','BAC']

# Define the time period
start_date = '2007-01-01'
end_date = '2022-01-01'
# Define the training period
#250 trading days (~1 year)
training_period = 250

import pandas as pd
import yfinance as yf
from scipy.stats import genpareto
import numpy as np
import warnings
from scipy.stats import genextreme
import matplotlib.pyplot as plt
import scipy.stats as stats

# Download stock data from yahoo finance using the define tickers and time period
def download_asset_data(ticker, date_start, date_end):
  stock_prices = yf.download(ticker, start=date_start, end=date_end)['Adj Close']
  #cap_data.reset_index(inplace=True)
  return stock_prices

def value_at_risk(returns, confidence_level):
    """
    Compute the Value-at-Risk metric of returns at confidence_level
    :param returns: DataFrame
    :param confidence_level: float
    :return: float
    """

    # Calculate the highest return in the lowest quantile (based on confidence level)
    var = returns.quantile(q=confidence_level, interpolation="higher")
    return var


def expected_shortfall(returns, confidence_level):
    """
    Compute the Value-at-Risk metric of returns at confidence_level
    :param returns: DataFrame
    :param confidence_level: float
    :return: float
    """

    # Calculate the VaR of the returns
    var = value_at_risk(returns, confidence_level)
    # Find all returns in the worst quantitle
    worst_returns = returns[returns.lt(var)]
    # Calculate mean of all the worst returns
    es = worst_returns.mean()

    return es

# Function to compute EVT-Based VaR
def evt_value_at_risk(returns, confidence_level):
    """
    Compute the Value-at-Risk using Extreme Value Theory at a given confidence level.
    :param returns: DataFrame or Series of returns
    :param confidence_level: float (e.g., 0.99 for 99% confidence)
    :return: float
    """
    # Fit the GEV distribution to the negative returns
    neg_returns = -returns
    shape, loc, scale = genextreme.fit(neg_returns)

    # Calculate the quantile for the given confidence level
    evt = -genextreme.ppf(1 - confidence_level, shape, loc=loc, scale=scale)

    return evt

# Time Series Plot for the Returns of all the Assets
def timeseries_plot(stock_daily_return):
  # Time Series Plot for the Returns of all the Assets
  plt.figure(figsize=(12, 7))
  for column in stock_daily_return.columns:
      plt.plot(stock_daily_return.index, stock_daily_return[column], label=column)
  plt.title('Time Series Plot of Asset Returns')
  plt.xlabel('Date')
  plt.ylabel('Assets Returns')
  plt.legend()
  return plt.show()

stock_prices = download_asset_data(tickers, start_date, end_date)

# Calculate the 1-day returns for training the VaR models
daily_returns = stock_prices.pct_change(1, fill_method=None).dropna()
timeseries_plot(daily_returns)

# Download stock prices, compute VaR and ES and output result for each ticker
def process_ticker(ticker, training_start, testing_end_date):
    # Download training data
    stock_prices = download_asset_data(ticker, training_start, testing_end_date)

    # Calculate the 1-day returns for training the VaR models
    daily_returns = stock_prices.pct_change(1, fill_method=None).dropna()

    # Calculate the 10-days returns for training the ES models
    ten_days_returns = stock_prices.pct_change(10, fill_method=None).dropna()


    # Initialize a list to collect results
    results = []

    # Define the columns for the result DataFrame
    columns = [
        'date', 'ticker', 'var_990', 'es', 'evt', 'testing_data',
        'exceeds_var_990', 'exceeds_es', 'exceed_evt'
    ]

    # Loop over the range of days to calculate VaR and ES
    for day in range(1, len(ten_days_returns)):
        testing_data = daily_returns.iloc[-day]
        var_training_data = daily_returns.iloc[:-day].tail(training_period)
        es_training_data = ten_days_returns.iloc[:-day].tail(training_period)
        evt_training_data = daily_returns.iloc[:-day].tail(training_period)

        # Calculate VaR at 99% confidence level
        var_990 = value_at_risk(var_training_data, 0.01)

        # Calculate Expected Shortfall at 95% confidence level
        es = expected_shortfall(es_training_data, 0.025)

        # Compute EVT-Based VaR
        confidence_level = 0.99
        var_evt = evt_value_at_risk(evt_training_data, confidence_level )

        date = daily_returns.index[-day]

        # Compile the result into a list
        result = {
            'date': date,
            'ticker': ticker,
            'var_990': var_990,
            'es': es,
            'evt': var_evt,
            'testing_data': testing_data,
            'exceeds_var_990': testing_data > var_990,
            'exceeds_es': testing_data > es,
            'exceed_evt': testing_data > var_evt
        }

        # Append the result to the list
        results.append(result)

    # Convert the list of results to a DataFrame
    df_result = pd.DataFrame(results, columns=columns)

    # Save the result as a CSV file
    csv_filename = f'{ticker}_result.csv'
    df_result.to_csv(csv_filename, index=False)
    print(f'{ticker} done')
    return df_result

# Main
training_start_date = start_date
testing_date_plus_one = end_date
for ticker in tickers:
    result = process_ticker(ticker, training_start_date, testing_date_plus_one)
