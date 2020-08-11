# libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Data
#labels = [0.1, 0.01, 0.0]
#val_loss = [9.6921, 9.6827, 9.6768]
#training_time = [85.6, 93.2 , 120.2]
labels = [1, 2, 3, 4]
val_loss = [9.7776, 9.6827, 9.6618, 9.6600]
training_time = [18.6, 26.4, 33.7, 39.6]

x = np.arange(len(labels))
width = 0.35  # the width of the bars
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

rects1 = ax.bar(x - width/2, val_loss, width, color='blue', hatch="/", label = 'Validation Loss')
rects2 = ax2.bar(x + width/2, training_time, width, color='red', hatch="-", label='Training Time')

#ax.set_ylim([9.65,9.70])
#ax2.set_ylim([80, 125])
ax.set_ylim([9.60,9.80])
ax2.set_ylim([17, 41])

#ax.set_xlabel(r'$\epsilon$', fontsize=24)
ax.set_xlabel('K', fontsize=18)
ax.set_ylabel('Validation Loss', fontsize=14)
ax2.set_ylabel('Training Time/Epoch (seconds)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=2)
ax2.legend(loc=1)

fig.tight_layout()

plt.show()
