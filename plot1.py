# libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
import numpy as np
import pandas as pd
import seaborn as sns

# Data
x = range(1,13)
conv_lstm_pm25_rmse = [5.750224, 6.235271, 6.523195, 6.802026, 6.940228, 7.348467, 7.981356, 7.792602, 8.075660, 8.243859, 8.236531, 8.415257]
conv_lstm_pm25_mae = [3.812974, 4.073946, 4.374659, 4.687023, 4.909774, 5.207528, 5.381213, 5.620552, 5.745127, 5.967848, 6.228832, 6.267795]
conv_lstm_pm10_rmse = [13.851549, 14.467833, 14.719532, 15.454947, 15.839831, 16.284884, 16.876527, 17.150253, 17.841271, 17.792565, 18.170810, 18.098365]
conv_lstm_pm10_mae = [10.783310, 11.295587, 11.529755, 12.189555, 12.555805, 12.976139, 13.443876, 13.699309, 14.290650, 14.234435, 14.545069, 14.473372]

gcrnn_pm25_rmse = [5.288365,5.746339,6.155172,6.517802,6.845884,7.144789,7.418746,7.670674,7.903351,8.118986,8.319736,8.507352]
gcrnn_pm25_mae = [3.6720,3.9497,4.2016,4.4295,4.6391,4.8330,5.0132,5.1813,5.3386,5.4862,5.6250,5.7560]
gcrnn_pm10_rmse = [11.503078,12.226124,12.876906,13.465969,14.005359,14.500741,14.958274,15.382028,15.776203,16.143909,16.487661,16.809322]
gcrnn_pm10_mae = [6.4373,6.9527,7.4146,7.8302,8.2110,8.5619,8.8882,9.1926,9.4779,9.7459,9.9983,10.2360]

#df=pd.DataFrame({'x': x, 'y1': conv_lstm_pm25_rmse, 'y2': gcrnn_pm25_rmse, 'y3': conv_lstm_pm10_rmse, 'y4': gcrnn_pm10_rmse })
df=pd.DataFrame({'x': x, 'y1': conv_lstm_pm25_mae, 'y2': gcrnn_pm25_mae, 'y3': conv_lstm_pm10_mae, 'y4': gcrnn_pm10_mae })
# multiple line plot
sns.set()
plt.plot( 'x', 'y3', data=df, marker='o', markerfacecolor='red', markersize=8, color='red', linewidth=2, linestyle='dashed', label='ConvLSTM PM10')
plt.plot( 'x', 'y4', data=df, marker='^', markerfacecolor='red', markersize=8, color='red', linewidth=2, label='ST-GCRNN PM10')

plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=8, color='blue', linewidth=2, linestyle='dashed', label='ConvLSTM PM2.5')
plt.plot( 'x', 'y2', data=df, marker='^', markerfacecolor='blue', markersize=8, color='blue', linewidth=2, label='ST-GCRNN PM2.5')

plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d hours'))
plt.legend()
plt.xlabel("Forecasting Time Steps")
#plt.ylabel("Testing RMSE")
plt.ylabel("Testing MAE")
plt.show()
