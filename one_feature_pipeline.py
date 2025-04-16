import argparse
import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def main(data_path):
    df = pd.read_csv(data_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df[['Power']]
    df = np.array(df.values)
    
    n_steps = 52
    train, dev, test = np.split(df, [int(len(df) * 0.6), int(len(df) * 0.8)])
    X_train, y_train = split_sequence(train, n_steps)
    X_val, y_val = split_sequence(dev, n_steps)
    X_test, y_test = split_sequence(test, n_steps)
    
    scaler = MinMaxScaler(feature_range=(-1, 0))
    y_train = scaler.fit_transform(y_train)
    y_val = scaler.transform(y_val)
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(n_steps, 1)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    print(model.summary())
    model.fit(X_train, y_train, batch_size=1, epochs=5, validation_data=(X_val, y_val))
    
    final_preds = scaler.inverse_transform(model.predict(X_test))
    score_rmse = np.sqrt(mean_squared_error(y_test, final_preds[:, 0]))
    score_mae = mean_absolute_error(y_test, final_preds[:, 0])
    
    print(f'RMSE: {score_rmse}')
    print(f'MAE: {score_mae}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LSTM model for time series forecasting.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file.')
    args = parser.parse_args()
    main(args.data_path)