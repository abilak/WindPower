from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# function to preprocess data for CNN and LSTM (may require different steps, if so, make diff functions for each)

'''
FROM THE README OF THE DATASET (MAY BE HELPFUL):
The columns in the data are as follows:

Time - Hour of the day when readings occurred
temperature_2m - Temperature in degrees Fahrenheit at 2 meters above the surface
relativehumidity_2m - Relative humidity (as a percentage) at 2 meters above the surface
dewpoint_2m - Dew point in degrees Fahrenheit at 2 meters above the surface
windspeed_10m - Wind speed in meters per second at 10 meters above the surface
windspeed_100m - Wind speed in meters per second at 100 meters above the surface
winddirection_10m - Wind direction in degrees (0-360) at 10 meters above the surface (see notes)
winddirection_100m - Wind direction in degrees (0-360) at 100 meters above the surface (see notes)
windgusts_10m - Wind gusts in meters per second at 100 meters above the surface
Power - Turbine output, normalized to be between 0 and 1 (i.e., a percentage of maximum potential output)

Notes:
	1) Likely many of these variables will not be very relevant. They are included here but do not need to be included in the final models.
	2) Degrees are measured from 0 to 360. Since 0 and 360 represent the same spot on a circle, consider transforming these using sine and/or cosine. Also consider converting them to radians, instead of degrees.
	3) Each location can have a different model. There is no reason to build one model to work for all locations.
'''
'''
SEASONAL FEATURES - temperature_2m, dewpoint_2m
'''

# Lots of data preprocessing to be done - fourier features, lag features (partial autocorr), time dummies, etc.
# good place to look: https://www.kaggle.com/learn/time-series. not great tbh to learn models, but good for data processing

df = pd.read_csv('data/Location1.csv')
feature = 'windgusts_10m'
df = df[['Time', feature]]
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)
df = df.resample('W').mean()
plt.figure(figsize=(10, 6))
plt.plot(df.index, df[feature], marker='o', linestyle='-', color='b')
plt.title('Timestamp vs Target Variable')
plt.xlabel('Timestamp')
plt.ylabel('Target')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
def preprocess_data_cnn(data, feature_selection = 5): 
    '''Input as csv'''
    data = data.drop('Timestamp', axis = 1) # we
    data['winddirection_10m'] = np.cos(data['winddirection_10m'] * np.pi / 180) # maybe get rid of cosine transformation
    data['winddirection_100m'] = np.cos(data['winddirection_100m'] * np.pi / 180)
    target = data['Power']
    rest = data.drop('Power')
    # normalize - now that i think about it... we should be cautious on scaling (do research)
    scaler = MinMaxScaler()
    cols = target.columns 
    rest = scaler.fit_transform(rest)
    rest = pd.DataFrame(rest, columns = cols)
    
    # select best features 
    select_k_best = SelectKBest(score_func=chi2, k=feature_selection)
    rest = select_k_best.fit_transform(rest, target)
    
    
    
