from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('data/Location1.csv')
df['Time'] = pd.to_datetime(df['Time'])
df = df[['Power']]
df = np.array(df.values)
print(df)
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

n_steps = 52
train, dev, test = np.split(df, [int(len(df) * 0.6), int(len(df) * 0.8)])
X_train, y_train = split_sequence(train, n_steps)
X_val, y_val = split_sequence(dev, n_steps)
X_test, y_test = split_sequence(test, n_steps)
scaler = MinMaxScaler(feature_range=(-1, 0)) 
y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)
y_test = scaler.transform(y_test)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(n_steps, 1)))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 3e-4), loss='mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 15, batch_size = 1)

test_predict = scaler.inverse_transform(model.predict(X_test))
y_test = scaler.inverse_transform(y_test)
score_rmse = np.sqrt(mean_squared_error(y_test, test_predict[:,0]))
score_mae = mean_absolute_error(y_test, test_predict[:,0])
print('RMSE: {}'.format(score_rmse))