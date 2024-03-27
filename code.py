import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, N):
    dt = T/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt) # Cumulative sum to generate Wiener process
    S = S0*np.exp((mu - 0.5*sigma**2)*t + sigma*W)
    return t, S

# Parameters
S0 = 100  # Initial stock price
mu = 0.05  # Expected return
sigma = 0.2  # Volatility
T = 1.0  # Time period (1 year)
N = 365  # Number of steps

# Simulate GBM
t, S = simulate_gbm(S0, mu, sigma, T, N)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, S)
plt.title('Stock Price Simulation using GBM')
plt.xlabel('Time (Years)')
plt.ylabel('Stock Price')
plt.show()
