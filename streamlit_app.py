import streamlit as st
import numpy as np
import scipy
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

def black_scholes_option_pricer(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price

st.title('Black-Scholes Option Pricer by Guillermo Sanchez Garcia')

# User inputs
S = st.number_input('Stock Price (S)', value=100.0)
K = st.number_input('Strike Price (K)', value=100.0)
T = st.number_input('Time to Expiry (T) in years', value=1.0)
r = st.number_input('Risk-Free Interest Rate (r)', value=0.05)
sigma = st.number_input('Volatility (σ)', value=0.2)

if st.button('Calculate'):
    call, put = black_scholes_option_pricer(S, K, T, r, sigma)
    st.write(f"Call Price: {call}")
    st.write(f"Put Price: {put}")


# Heatmap visualization function
def plot_heatmap(S_range, sigma_range, K, T, r):
    call_prices = []
    put_prices = []
    for S in S_range:
        call_row = []
        put_row = []
        for sigma in sigma_range:
            call, put = black_scholes_option_pricer(S, K, T, r, sigma)
            call_row.append(call)
            put_row.append(put)
        call_prices.append(call_row)
        put_prices.append(put_row)

    # Plot call prices heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(call_prices, xticklabels=np.round(sigma_range, 2), yticklabels=np.round(S_range, 2), 
                cmap='RdYlGn', annot=True, fmt=".2f", cbar_kws={'label': 'Call Price'})
    plt.title('Call Prices Heatmap')
    plt.xlabel('Volatility (σ)')
    plt.ylabel('Stock Price (S)')
    st.pyplot(plt)

    # Plot put prices heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(put_prices, xticklabels=np.round(sigma_range, 2), yticklabels=np.round(S_range, 2), 
                cmap='RdYlGn', annot=True, fmt=".2f", cbar_kws={'label': 'Put Price'})
    plt.title('Put Prices Heatmap')
    plt.xlabel('Volatility (σ)')
    plt.ylabel('Stock Price (S)')
    st.pyplot(plt)



# Add heatmap to Streamlit app
if st.button('Show Heatmap'):
    S_range = np.linspace(S * 0.5, S * 1.5, 10)
    sigma_range = np.linspace(sigma * 0.5, sigma * 1.5, 10)
    plot_heatmap(S_range, sigma_range, K, T, r)

