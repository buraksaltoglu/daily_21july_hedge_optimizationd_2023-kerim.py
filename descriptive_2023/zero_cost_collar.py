import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the data
maturities = [30, 60, 90, 180, 270, 360]
moneyness = ['c10', 'c15', 'c25', 'c30', 'ATM', 'p65', 'p75', 'p85', 'p90']
x, y = np.meshgrid(range(len(maturities)), range(len(moneyness)))

# Define the volatility values for each combination of maturities and moneyness
volatility = np.array([
    [36.23233001, 35.05420909, 34.27321214, 34.1909774, 34.74019277, 36.03375591, 37.4982588, 40.11198624, 42.53786818],
    [35.22565687, 34.3231005, 33.7545488, 33.77091928, 34.38941687, 35.70666169, 37.18597441, 39.74207906, 42.01992528],
    [34.58493085, 33.7658656, 33.25688682, 33.34547857, 34.05450783, 35.4469146, 36.96020373, 39.50585182, 41.702226],
    [33.14837517, 32.38911126, 31.8662354, 31.98717957, 32.92137575, 34.54823525, 36.17308994, 38.74315232, 40.81030246],
    [31.52553304, 30.85449786, 30.39939721, 30.53795369, 31.56766102, 33.33368326, 35.05138635, 37.62353271, 39.63395167],
    [30.30366905, 29.58981177, 29.1515893, 29.37237523, 30.12689387, 31.90302876, 33.60727655, 36.14988639, 38.14935796]
])

# Increase plot size
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
ax.plot_surface(x.T, y.T, volatility, cmap='viridis', alpha=0.5)

# Increase tick label font size
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='z', labelsize=14)

# Set labels and title for the surface plot
ax.set_xlabel('Maturity', fontsize=15)
ax.set_ylabel('Moneyness', fontsize=15)
ax.set_zlabel('Volatility', fontsize=15)
ax.set_title('Volatility Surface', fontsize=17)

# Set the custom tick labels for the x-axis and y-axis
ax.set_xticks(range(len(maturities)))
ax.set_xticklabels(maturities)
ax.set_yticks(range(len(moneyness)))
ax.set_yticklabels(moneyness)

# Create scatter plot points with different colors
scatter_x = np.random.randint(0, len(maturities), size=50)
scatter_y = np.random.randint(0, len(moneyness), size=50)
scatter_z = np.random.uniform(30, 42, size=50)
scatter_colors = np.random.rand(50)

ax.scatter(scatter_x, scatter_y, scatter_z, c=scatter_colors, cmap='coolwarm')

# Set labels and title for the scatter plot
ax.set_xlabel('Maturity', fontsize=16)
ax.set_ylabel('Moneyness', fontsize=16)
ax.set_zlabel('Volatility', fontsize=16)
ax.set_title('Scatter Plot', fontsize=16)

# Show the plot
plt.show()
