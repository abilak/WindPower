from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess
import tensorflow as tf 
from sklearn.model_selection import train_test_split
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
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

df = np.array(df.values)
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

n_steps = 52
train, dev, test = np.split(df, [int(len(df) * 0.6), int(len(df) * 0.8)])

X_train, y_train = train[:,:-1], train[:, -1]
X_val, y_val = dev[:,:-1], dev[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]
scaler = MinMaxScaler() 
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
train_data, dev_data, test_data = np.hstack((X_train, y_train.reshape((-1, 1)))), np.hstack((X_val, y_val.reshape((-1, 1)))), np.hstack((X_test, y_test.reshape((-1, 1))))
X_train, y_train = split_sequences(train_data, n_steps)
X_val, y_val = split_sequences(dev_data, n_steps)
X_test, y_test = split_sequences(test_data, n_steps)

'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(n_steps, 8)))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 3e-4), loss='mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 50, batch_size = 200)
'''

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(n_steps, 8)))
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dense(25))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 3e-4), loss='mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError()])
print(model.summary())
#Train the model
model.fit(X_train, y_train, batch_size=1, epochs=5, validation_data=(X_val, y_val))

preds = model.predict(X_train)
y_train = y_train-preds 

model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(n_steps, 8)))
model2.add(tf.keras.layers.LSTM(64, return_sequences=False))
model2.add(tf.keras.layers.Dense(25))
model2.add(tf.keras.layers.Dense(1))
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 3e-4), loss='mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError()])
print(model2.summary())

model2.fit(X_train, y_train, batch_size=1, epochs=5)

final_preds = model.predict(X_test) + model2.predict(X_test)

print()