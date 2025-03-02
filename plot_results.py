import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt

# Data from the table
electrolyzer_power = [0, 25, 35, 42, 45, 50, 75, 100, 150]
npv_over_capex_ptg = [1.327, 1.447, 1.466, 1.470, 1.471, 1.468, 1.410, 1.390, 1.095]

grid_connection = [0, 50, 100, 150, 200, 250, 300, 350]
npv_over_capex_grid = [0.164, 0.647, 1.066, 1.282, 1.342, 1.410, 1.471, 1.439]

# Plot for Electrolyzer Power
plt.figure(figsize=(10, 5))
plt.plot(electrolyzer_power, npv_over_capex_ptg, marker='o', linestyle='-', color='b', label="NPV/CAPEX")
plt.xlabel("Electrolyzer Power")
plt.ylabel("NPV/CAPEX")
plt.title("Sensitivity Analysis: NPV/CAPEX vs Electrolyzer Power")
plt.grid(True)
plt.legend()
plt.show()

# Plot for Grid Connection
plt.figure(figsize=(10, 5))
plt.plot(grid_connection, npv_over_capex_grid, marker='o', linestyle='-', color='r', label="NPV/CAPEX")
plt.xlabel("Grid Connection")
plt.ylabel("NPV/CAPEX")
plt.title("Sensitivity Analysis: NPV/CAPEX vs Grid Connection")
plt.grid(True)
plt.legend()
plt.show()



