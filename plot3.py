# libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Data
labels = ['1 hour', '6 hours', '12 hours']
rmse_1 = [5.3320, 7.1841, 8.5481]
rmse_6 = [5.3240, 7.1806, 8.5434]
rmse_12 = [5.3091, 7.1717, 8.5367]

x = np.arange(len(labels))
width = 0.2  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x-0.3, rmse_1, width, hatch="-",  label="Spectral Convolution")
rects2 = ax.bar(x-0.1, rmse_6, width, hatch="/",  label="Random Walk Diffusion Convolution")
rects32 = ax.bar(x+0.1, rmse_12, width, hatch="\\",  label="Dual Random Walk Diffusion Convolution")

axes = plt.gca()
axes.set_ylim([5.25, 8.60])

ax.set_xlabel('Time steps', fontsize=14)
ax.set_ylabel('RMSE', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
