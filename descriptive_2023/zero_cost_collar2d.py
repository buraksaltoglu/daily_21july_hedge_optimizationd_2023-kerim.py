import numpy as np
import matplotlib.pyplot as plt

# Define the data
maturity = 30
moneyness = ['c30', 'ATM', 'p65', 'p75', 'p85', 'p90']

volatility = np.array([
    [36.23233001, 35.05420909, 34.27321214, 34.1909774, 34.74019277, 36.03375591, 37.4982588, 40.11198624, 42.53786818],
    [35.22565687, 34.3231005, 33.7545488, 33.77091928, 34.38941687, 35.70666169, 37.18597441, 39.74207906, 42.01992528],
    [34.58493085, 33.7658656, 33.25688682, 33.34547857, 34.05450783, 35.4469146, 36.96020373, 39.50585182, 41.702226],
    [33.14837517, 32.38911126, 31.8662354, 31.98717957, 32.92137575, 34.54823525, 36.17308994, 38.74315232, 40.81030246],
    [31.52553304, 30.85449786, 30.39939721, 30.53795369, 31.56766102, 33.33368326, 35.05138635, 37.62353271, 39.63395167],
    [30.30366905, 29.58981177, 29.1515893, 29.37237523, 30.12689387, 31.90302876, 33.60727655, 36.14988639, 38.14935796]
])

# Filter the volatility values for the specified maturity
filtered_volatility = volatility[:, moneyness.index(str(maturity))]

# Create the figure and subplot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the volatility for the selected maturity
ax.plot(moneyness, filtered_volatility, marker='o', label=f'{maturity} days')

# Set labels and title
ax.set_xlabel('Moneyness')
ax.set_ylabel('Volatility')
ax.set_title(f'Volatility by Moneyness (Maturity: {maturity} days)')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# Show the plot
plt.show()
