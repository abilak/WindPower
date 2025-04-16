import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from torchdyn.models import NeuralODE
from torchdyn.core import ODEProblem
from torchdyn.numerics import odeint
import matplotlib.pyplot as plt

class RNNODEFunc(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
        self.rnn_cells = nn.ModuleList([
            nn.GRUCell(1 if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.activation1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, t, x):
        batch_size = x.shape[0]
        
        if not hasattr(self, 'hidden_states'):
            self.hidden_states = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        
        h = x
        for i in range(self.num_layers):
            self.hidden_states[i] = self.rnn_cells[i](
                h if i == 0 else self.hidden_states[i-1], 
                self.hidden_states[i]
            )
            
            if i < self.num_layers - 1:
                self.hidden_states[i] = self.dropout(self.hidden_states[i])
        h_final = self.hidden_states[-1]
        h_norm = self.layer_norm(h_final)
        h1 = self.activation1(self.linear1(h_norm))
        h2 = self.activation2(self.linear2(h1))
        dx = self.output(h2)
        return dx
    def reset_hidden(self, batch_size, device):
        self.hidden_states = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

class NeuralODEModel(pl.LightningModule):
    def __init__(self, input_size=1, hidden_size=64, forecast_horizon=10, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.ode_func = RNNODEFunc(hidden_size=hidden_size, num_layers=2, dropout=0.2)
        self.ode_problem = ODEProblem(self.ode_func, solver='dopri5', rtol=1e-3, atol=1e-4)
        self.node = NeuralODE(self.ode_problem)
        self.output_layer = nn.Linear(input_size, forecast_horizon)
        self.loss_fn = nn.MSELoss()
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        self.ode_func.reset_hidden(batch_size, x.device)
        x0 = x[:, -1, :]
        if hasattr(self.ode_func, 'h'):
            delattr(self.ode_func, 'h')
        t_span = torch.linspace(0, 1, 2)
        trajectory = odeint(self.ode_func, x0, t_span, solver='dopri5')[1]
        print(f"Trajectory shape: {trajectory.shape}")
        final_state = trajectory[-1, :, :] 
        print(f"Final state shape: {final_state.shape}")
        forecast = self.output_layer(final_state)
        print(f"Forecast shape: {forecast.shape}")
        return forecast
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  
        if y.shape != y_hat.shape:
            print(f"Shape mismatch - y: {y.shape}, y_hat: {y_hat.shape}")
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, data_path, sequence_length=24, forecast_horizon=10, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
    def setup(self, stage=None):
        df = pd.read_csv(self.data_path)
        power_data = df['Power'].values.reshape(-1, 1)
        X, y = self._create_sequences(power_data)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        self.train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        self.val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        self.test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_test)
        )
    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[(i + self.sequence_length):(i + self.sequence_length + self.forecast_horizon), 0])
        return np.array(X), np.array(y)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

def evaluate_model(model, data_module):
    model.eval()
    device = next(model.parameters()).device
    
    all_y_true = []
    all_y_pred = []
    
    test_loader = data_module.test_dataloader()
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_pred = model(x).cpu().numpy()
            all_y_true.append(y.numpy())
            all_y_pred.append(y_pred)
    
    if all_y_true and all_y_pred:
        flat_y_true = []
        flat_y_pred = []
        
        for batch_true, batch_pred in zip(all_y_true, all_y_pred):
            for sample_true, sample_pred in zip(batch_true, batch_pred):
                flat_y_true.append(sample_true)
                flat_y_pred.append(sample_pred)
        
        all_y_true = np.array(flat_y_true)
        all_y_pred = np.array(flat_y_pred)
        
        horizon_metrics = []
        for h in range(all_y_true.shape[1]):
            y_true_h = all_y_true[:, h]
            y_pred_h = all_y_pred[:, h]
            
            mse_h = mean_squared_error(y_true_h, y_pred_h)
            rmse_h = sqrt(mse_h)
            
            horizon_metrics.append({
                'horizon': h+1,
                'mse': mse_h,
                'rmse': rmse_h
            })
        
        overall_mse = mean_squared_error(all_y_true.flatten(), all_y_pred.flatten())
        overall_rmse = sqrt(overall_mse)
        
        results = {
            'overall': {
                'mse': overall_mse,
                'rmse': overall_rmse
            },
            'by_horizon': horizon_metrics
        }
    else:
        results = {
            'overall': {
                'mse': float('nan'),
                'rmse': float('nan')
            },
            'by_horizon': []
        }
    
    return results

def train_neural_ode():
    pl.seed_everything(42)
    
    SEQUENCE_LENGTH = 52
    FORECAST_HORIZON = 1
    BATCH_SIZE = 32
    HIDDEN_SIZE = 64
    MAX_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    data_module = TimeSeriesDataModule(
        data_path='./data/Location4.csv',
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        batch_size=BATCH_SIZE
    )
    
    model = NeuralODEModel(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        forecast_horizon=FORECAST_HORIZON,
        lr=LEARNING_RATE
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='neural_ode-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    trainer.fit(model, data_module)
    
    trainer.test(model, data_module)
    
    return model, data_module

model, data_module = train_neural_ode()

evaluation_results = evaluate_model(model, data_module)

print("\nMODEL EVAL RESULTS:")
print(f"MSE: {evaluation_results['overall']['mse']:.6f}")
print(f"RMSE: {evaluation_results['overall']['rmse']:.6f}")

print("Results by Forecast Horizon:")
print("Horizon\tMSE\t\tRMSE")
for horizon_result in evaluation_results['by_horizon']:
    h = horizon_result['horizon']
    mse = horizon_result['mse']
    rmse = horizon_result['rmse']
    print(f"{h}\t{mse:.6f}\t{rmse:.6f}")