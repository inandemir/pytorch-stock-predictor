import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg') # Non-interactive backend to prevent UI threading issues
import matplotlib.pyplot as plt

# Define the LSTM Model Class
class StockLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden and cell states to zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Select last time step's output
        out = self.fc(out[:, -1, :])
        return out

class StockPredictorEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = None
        self.lookback = 20
        self.last_prices = [] # Store last prices of selected stock for prepopulating
        
    def list_stocks(self):
        """Lists available stock CSV files in the data directory."""
        if not os.path.exists(self.data_dir):
            return []
        files = os.listdir(self.data_dir)
        stocks = []
        for f in files:
            if f.endswith(".csv") and "all_stocks" not in f:
                stocks.append({
                    "filename": f,
                    "name": f.split("_")[0]
                })
        return stocks

    def prepare_data(self, filepath):
        """Loads and processes stock data from a CSV file."""
        df = pd.read_csv(filepath)
        # Sort by Date if present
        if "Date" in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
        # Get Close price
        close_prices = df['Close'].values.astype(float).reshape(-1, 1)
        self.last_prices = close_prices[-self.lookback:].flatten().tolist()
        
        # Scale data
        scaled_data = self.scaler.fit_transform(close_prices)
        
        # Create sliding sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback):
            X.append(scaled_data[i:i + self.lookback])
            y.append(scaled_data[i + self.lookback])
            
        X = np.array(X)
        y = np.array(y)
        
        # Split into train / test (80% / 20%)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Get real dates for test set sequences
        dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
        valid_dates = dates[self.lookback:]
        test_dates = valid_dates[split:]
        
        return X_train, X_test, y_train, y_test, close_prices, split, test_dates

    def train(self, stock_filename, progress_callback=None, epochs=25):
        """Trains the LSTM model on a specific stock CSV."""
        filepath = os.path.join(self.data_dir, stock_filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Stock data file not found: {filepath}")
            
        X_train, X_test, y_train, y_test, raw_prices, split_idx, test_dates = self.prepare_data(filepath)
        
        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        
        # Create Model
        self.model = StockLSTM(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            if progress_callback:
                progress_callback(epoch + 1, epochs, loss_val)
                
        # Evaluate model on test set
        self.model.eval()
        with torch.no_grad():
            test_predictions = self.model(X_test_t).numpy()
            
        # Inverse transform to original prices
        predictions_unscaled = self.scaler.inverse_transform(test_predictions)
        y_test_unscaled = self.scaler.inverse_transform(y_test)
        
        # Calculate Metrics
        mse = np.mean((y_test_unscaled - predictions_unscaled) ** 2)
        rmse = np.sqrt(mse)
        
        # Save trained weights
        model_save_dir = os.path.dirname(filepath)
        weights_path = os.path.join(model_save_dir, f"{stock_filename.split('_')[0]}_lstm.pth")
        torch.save(self.model.state_dict(), weights_path)
        
        # Generate and save Matplotlib plot (fallback)
        self.save_chart(y_test_unscaled, predictions_unscaled, stock_filename.split('_')[0])
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "weights_path": weights_path,
            "last_prices": self.last_prices,
            "test_dates": test_dates,
            "actual_prices": y_test_unscaled.flatten().tolist(),
            "predicted_prices": predictions_unscaled.flatten().tolist()
        }

    def save_chart(self, actual, predicted, stock_name):
        """Generates actual vs predicted stock chart using matplotlib."""
        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')
        plt.plot(actual, label='Gerçek Fiyatlar', color='#0078d4', linewidth=2)
        plt.plot(predicted, label='Yapay Zeka Tahmini', color='#00bcf2', linestyle='--', linewidth=2)
        plt.title(f"{stock_name} Hisse Senedi Fiyat Tahmini - LSTM Test Sonuçları", fontsize=14, pad=15)
        plt.xlabel("Günler (Test Seti)", fontsize=11)
        plt.ylabel("Kapanış Fiyatı (USD)", fontsize=11)
        plt.legend(frameon=True, facecolor=(0, 0, 0, 0.5), edgecolor=(1, 1, 1, 0.1))
        plt.grid(color=(1, 1, 1, 0.05), linestyle='-')
        
        # Ensure static folder exists in local_rag_app
        static_dir = os.path.join(os.path.dirname(self.data_dir), "static")
        os.makedirs(static_dir, exist_ok=True)
        chart_path = os.path.join(static_dir, "stock_chart.png")
        
        plt.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close()

    def predict_next(self, prices_list):
        """Predicts the next day's price given a list of lookback closing prices."""
        if not self.model:
            raise ValueError("Model is not trained yet. Train a model first.")
        if len(prices_list) != self.lookback:
            raise ValueError(f"Input prices list must contain exactly {self.lookback} values.")
            
        # Scale input
        prices_arr = np.array(prices_list).reshape(-1, 1)
        scaled_input = self.scaler.transform(prices_arr)
        
        # Prepare tensor [batch_size=1, sequence_length=lookback, features=1]
        input_t = torch.tensor(scaled_input.reshape(1, self.lookback, 1), dtype=torch.float32)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            scaled_prediction = self.model(input_t).numpy()
            
        # Inverse scale
        prediction = self.scaler.inverse_transform(scaled_prediction)
        return float(prediction[0][0])

    def get_stock_dates(self, stock_filename):
        """Returns list of available dates for a stock, starting from index 20."""
        filepath = os.path.join(self.data_dir, stock_filename)
        if not os.path.exists(filepath):
            return []
            
        df = pd.read_csv(filepath)
        if "Date" not in df.columns:
            return []
            
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # We need at least lookback days before any date we predict
        dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
        if len(dates) <= self.lookback:
            return []
            
        valid_dates = dates[self.lookback:]
        valid_dates.reverse() # Show newest first
        return valid_dates

    def get_stock_history_by_date(self, stock_filename, selected_date):
        """Returns the 20 closing prices prior to selected_date and the actual price of selected_date."""
        filepath = os.path.join(self.data_dir, stock_filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Stock file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Find index of selected date
        matches = df[df['Date_Str'] == selected_date]
        if matches.empty:
            raise ValueError(f"Date {selected_date} not found in stock data.")
            
        idx = df.index[df['Date_Str'] == selected_date][0]
        if idx < self.lookback:
            raise ValueError(f"Not enough history before date {selected_date}.")
            
        # Extract 20 days prior close prices
        prior_rows = df.iloc[idx - self.lookback : idx]
        prior_prices = prior_rows['Close'].values.astype(float).tolist()
        actual_price = float(df.iloc[idx]['Close'])
        
        return prior_prices, actual_price
