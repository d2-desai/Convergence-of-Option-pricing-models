#!/usr/bin/env python
# coding: utf-8

# # Binomial Option Pricing Model with Accuracy

# In[1]:


import numpy as np
import yahoo_fin as yf
import yahoo_fin.stock_info as si
from yahoo_fin import options


# In[2]:


def multistep_binomial_option_price(S, K, r, T, sigma, n, option_type):
    """
    Implements the multistep binomial option pricing model to calculate the price of a European option.

    Args:
        S (float): The current price of the underlying asset.
        K (float): The strike price of the option.
        r (float): The risk-free interest rate.
        T (float): The time to expiration of the option, in years.
        sigma (float): The volatility of the underlying asset.
        n (int): The number of time steps in the model.
        option_type (str): The type of the option, either 'call' or 'put'.

    Returns:
        The price of the option.
    """
    
    dt = T / n
    u = np.exp(sigma * np.sqrt(2*dt))
    d = 1 / u
    p = (np.exp(r * dt/2) - d) / (u - d)
    q = 1 - p
    
    # Generate the stock price tree
    stock_tree = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            stock_tree[i, j] = S * u**(i-j) * d**j
            
    # Generate the option value tree
    option_tree = np.zeros((n+1, n+1))
    if option_type == 'put':
        option_tree[:, n] = np.maximum(np.zeros(n+1), stock_tree[:, n] - K)
    else:
        option_tree[:, n] = np.maximum(np.zeros(n+1), K - stock_tree[:, n])
        
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            option_tree[i, j] = np.exp(-r*dt) * (p*option_tree[i+1,j] + q*option_tree[i+1,j+1])
            
    return option_tree[0, 0]

# List of stock symbols
symbols = ['AAPL', 'GOOG', 'AMZN', 'TSLA']


# Define option parameters
K = 4  # Strike price
r = 0.02  # Risk-free interest rate
T = 1  # Time to expiration of the option, in years
n = 100  # Number of time steps in the binomial model



for symbol in symbols:
    # Load market data from Yahoo Finance for each symbol separately
    data = si.get_data(symbol, start_date='2022-03-01', end_date='2022-03-31')['adjclose']
    S = data[-1]  # Current stock price
    sigma = np.std(data.pct_change()) * np.sqrt(252)  # Volatility of the stock
    option_type = 'call'
    price = multistep_binomial_option_price(S, K, r, T, sigma, n, option_type)
    print(symbol, 'call option price:', price)


# In[3]:


# Get real-time option prices from Yahoo Finance
option_prices = {}
for symbol in symbols:
    option_chain = options.get_options_chain(symbol)
    call_options = option_chain['puts']
    for i in range(len(call_options)):
        option = call_options.iloc[i]
        option_prices[(symbol, option['Contract Name'])] = option['Last Price']
for key in option_prices:
    print(key, 'actual price', option_prices[key])


# In[4]:


# Calculate model prices using the binomial option pricing model
model_prices = {}
for symbol in symbols:
    data = si.get_data(symbol, start_date='2022-03-01', end_date='2022-03-31')['adjclose']
    S = data[-1]  # Current stock price
    sigma = np.std(data.pct_change()) * np.sqrt(252)  # Volatility of the stock
    option_chain = options.get_options_chain(symbol)
    call_options = option_chain['puts']
    for i in range(len(call_options)):
        option = call_options.iloc[i]
        option_name = option['Contract Name']
        option_type = 'call'
        price = multistep_binomial_option_price(S, K, r, T, sigma, n, option_type)
        model_prices[(symbol, option_name)] = price
for key in model_prices:
    print(key, 'model price', model_prices[key])


# In[5]:


# Calculate RMSE for each option
rmse = {}
for key in option_prices:
    if key in model_prices:
        diff = option_prices[key] - model_prices[key]
        rmse[key] = np.sqrt(np.mean(diff**2))

# Print RMSE for each option
for key in rmse:
    print(key, 'RMSE:', rmse[key])


# In[6]:


# Calculate accuracy for each option
accuracy = {}
for key in rmse:
    if key in option_prices:
        avg_price = np.mean(option_prices[key])
        accuracy[key] = 1 - (rmse[key] / avg_price)

# Print accuracy for each option
for key in accuracy:
    print(key, 'accuracy:', accuracy[key])
    


# In[10]:


from enum import Enum
from abc import ABC, abstractclassmethod

class OPTION_TYPE(Enum):
    CALL_OPTION = 'Call Option'
    PUT_OPTION = 'Put Option'

class OptionPricingModel(ABC):
    """Abstract class defining interface for option pricing models."""

    def calculate_option_price(self, option_type):
        """Calculates call/put option price according to the specified parameter."""
        if option_type == OPTION_TYPE.CALL_OPTION.value:
            return self._calculate_call_option_price()
        elif option_type == OPTION_TYPE.PUT_OPTION.value:
            return self._calculate_put_option_price()
        else:
            return -1

    @abstractclassmethod
    def _calculate_call_option_price(self):
        """Calculates option price for call option."""
        pass

    @abstractclassmethod
    def _calculate_put_option_price(self):
        """Calculates option price for put option."""
        pass


# In[13]:


# Standard library imports
import datetime

# Third party imports
import requests_cache
import matplotlib.pyplot as plt
from pandas_datareader import data as wb


class Ticker:
    """Class for fetcing data from yahoo finance."""
    
    @staticmethod
    def get_historical_data(ticker, start_date=None, end_date=None, cache_data=True, cache_days=1):
        """
        Fetches stock data from yahoo finance. Request is by default cashed in sqlite db for 1 day.
        
        Params:
        ticker: ticker symbol
        start_date: start date for getting historical data
        end_date: end date for getting historical data
        cache_date: flag for caching fetched data into slqite db
        cache_days: number of days data will stay in cache 
        """
        try:
            # initializing sqlite for caching yahoo finance requests
            expire_after = datetime.timedelta(days=1)
            session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)

            # Adding headers to session
            session.headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0', 'Accept': 'application/json;charset=utf-8'}  # noqa
            
            if start_date is not None and end_date is not None:
                data = wb.DataReader(ticker, data_source='yahoo', start=start_date, end=end_date, session=session)
            else:
                data = wb.DataReader(ticker, data_source='yahoo', session=session)   #['Adj Close']
            if data is None:
                return None
            return data
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def get_columns(data):
        """
        Gets dataframe columns from previously fetched stock data.
        
        Params:
        data: dataframe representing fetched data
        """
        if data is None:
            return None
        return [column for column in data.columns]

    @staticmethod
    def get_last_price(data, column_name):
        """
        Returns last available price for specified column from already fetched data.
        
        Params:
        data: dataframe representing fetched data
        column_name: name of the column in dataframe
        """
        if data is None or column_name is None:
            return None
        if column_name not in Ticker.get_columns(data):
            return None
        return data[column_name].iloc[len(data) - 1]


    @staticmethod
    def plot_data(data, ticker, column_name):
        """
        Plots specified column values from dataframe.
        
        Params:
        data: dataframe representing fetched data
        column_name: name of the column in dataframe
        """
        try:
            if data is None:
                return
            data[column_name].plot()
            plt.ylabel(f'{column_name}')
            plt.xlabel('Date')
            plt.title(f'Historical data for {ticker} - {column_name}')
            plt.legend(loc='best')
            plt.show()
        except Exception as e:
            print(e)
            return


# In[16]:


# Third party imports
import numpy as np
from scipy.stats import norm 

# Local package imports
#from .base import OptionPricingModel


class BinomialTreeModel(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using BOPM (Binomial Option Pricing Model).
    It caclulates option prices in discrete time (lattice based), in specified number of time points between date of valuation and exercise date.
    This pricing model has three steps:
    - Price tree generation
    - Calculation of option value at each final node 
    - Sequential calculation of the option value at each preceding node
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps):
        """
        Initializes variables used in Black-Scholes formula .

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option cotract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        number_of_time_steps: number of time periods between the valuation date and exercise date
        """
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma
        self.number_of_time_steps = number_of_time_steps

    def _calculate_call_option_price(self): 
        """Calculates price for call option according to the Binomial formula."""
        # Delta t, up and down factors
        dT = self.T / self.number_of_time_steps                             
        u = np.exp(self.sigma * np.sqrt(dT))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(self.number_of_time_steps + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(self.S * u**j * d**(self.number_of_time_steps - j)) for j in range(self.number_of_time_steps + 1)])

        a = np.exp(self.r * dT)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   

        V[:] = np.maximum(S_T - self.K, 0.0)
    
        # Overriding option price 
        for i in range(self.number_of_time_steps - 1, -1, -1):
            V[:-1] = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1]) 

        return V[0]

    def _calculate_put_option_price(self): 
        """Calculates price for put option according to the Binomial formula."""  
        # Delta t, up and down factors
        dT = self.T / self.number_of_time_steps                             
        u = np.exp(self.sigma * np.sqrt(dT))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(self.number_of_time_steps + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(self.S * u**j * d**(self.number_of_time_steps - j)) for j in range(self.number_of_time_steps + 1)])

        a = np.exp(self.r * dT)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   

        V[:] = np.maximum(self.K - S_T, 0.0)
    
        # Overriding option price 
        for i in range(self.number_of_time_steps - 1, -1, -1):
            V[:-1] = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1]) 

        return V[0]


# In[19]:


"""
Script testing functionalities of option_pricing package:
- Testing stock data fetching from Yahoo Finance using pandas-datareader
- Testing Black-Scholes option pricing model   
- Testing Binomial option pricing model   
- Testing Monte Carlo Simulation for option pricing   
"""

#from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel, Ticker

# Fetching the prices from yahoo finance
data = Ticker.get_historical_data('TSLA')
print(Ticker.get_columns(data))
print(Ticker.get_last_price(data, 'Adj Close'))
Ticker.plot_data(data, 'TSLA', 'Adj Close')

# # Black-Scholes model testing
# BSM = BlackScholesModel(100, 100, 365, 0.1, 0.2)
# print(BSM.calculate_option_price('Call Option'))
# print(BSM.calculate_option_price('Put Option'))

# Binomial model testing
BOPM = BinomialTreeModel(100, 100, 365, 0.1, 0.2, 15000)
print(BOPM.calculate_option_price('Call Option'))
print(BOPM.calculate_option_price('Put Option'))

# # Monte Carlo simulation testing
# MC = MonteCarloPricing(100, 100, 365, 0.1, 0.2, 10000)
# MC.simulate_prices()
# print(MC.calculate_option_price('Call Option'))
# print(MC.calculate_option_price('Put Option'))
# MC.plot_simulation_results(20)


# In[ ]:




