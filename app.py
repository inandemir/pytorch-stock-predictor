import os
import threading
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from stock_predictor import StockPredictorEngine

app = Flask(__name__, static_folder="static")
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Initialize Stock Predictor
stock_predictor = StockPredictorEngine(data_dir=DATA_DIR)

# Global Training State
stock_training_state = {
    "status": "idle", # idle, training, ready, error
    "epoch": 0,
    "total_epochs": 25,
    "loss": 0.0,
    "metrics": None,
    "active_stock": None,
    "error": None
}

def bg_train_stock(stock_filename):
    global stock_training_state, stock_predictor
    try:
        stock_training_state["status"] = "training"
        stock_training_state["epoch"] = 0
        stock_training_state["loss"] = 0.0
        stock_training_state["active_stock"] = stock_filename.split("_")[0]
        stock_training_state["error"] = None
        stock_training_state["metrics"] = None
        
        def progress_callback(epoch, total, loss_val):
            stock_training_state["epoch"] = epoch
            stock_training_state["total_epochs"] = total
            stock_training_state["loss"] = loss_val
            
        metrics = stock_predictor.train(stock_filename, progress_callback=progress_callback, epochs=25)
        
        stock_training_state["metrics"] = metrics
        stock_training_state["status"] = "ready"
    except Exception as e:
        stock_training_state["status"] = "error"
        stock_training_state["error"] = str(e)

# Serve Frontend static files
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# 1. List Stocks
@app.route("/api/stock/list", methods=["GET"])
def get_stocks_list():
    try:
        stocks = stock_predictor.list_stocks()
        return jsonify({"stocks": stocks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2. Stock Training Status
@app.route("/api/stock/status", methods=["GET"])
def get_stock_status():
    return jsonify(stock_training_state)

# 3. Start Training
@app.route("/api/stock/train", methods=["POST"])
def train_stock_model():
    data = request.json or {}
    stock_filename = data.get("filename")
    if not stock_filename:
        return jsonify({"error": "filename is required"}), 400
        
    if stock_training_state["status"] == "training":
        return jsonify({"error": "A model is already training"}), 400
        
    thread = threading.Thread(target=bg_train_stock, args=(stock_filename,))
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": f"Started training for {stock_filename}", "state": stock_training_state})

# 4. Predict Next Close Price
@app.route("/api/stock/predict", methods=["POST"])
def predict_stock():
    data = request.json or {}
    prices = data.get("prices")
    if not prices or len(prices) != 20:
        return jsonify({"error": "Exactly 20 closing prices are required"}), 400
        
    try:
        prediction = stock_predictor.predict_next(prices)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 5. List Dates for Stock
@app.route("/api/stock/dates", methods=["GET"])
def get_stock_dates():
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "filename parameter is required"}), 400
    try:
        dates = stock_predictor.get_stock_dates(filename)
        return jsonify({"dates": dates})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 6. Fetch 20 days prior prices and actual price for Selected Date
@app.route("/api/stock/history", methods=["GET"])
def get_stock_history():
    filename = request.args.get("filename")
    date = request.args.get("date")
    if not filename or not date:
        return jsonify({"error": "filename and date parameters are required"}), 400
    try:
        prices, actual = stock_predictor.get_stock_history_by_date(filename, date)
        return jsonify({"prices": prices, "actual": actual})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 7. Download Stock Data from Yahoo Finance
@app.route("/api/stock/download", methods=["POST"])
def download_stock_data():
    data = request.json or {}
    ticker = data.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "Ticker sembolü gereklidir"}), 400
        
    try:
        # Download from 2018-01-01 to 2026-07-15
        df = yf.download(ticker, start="2018-01-01", end="2026-07-15")
        if df.empty:
            return jsonify({"error": f"{ticker} sembolü için veri bulunamadı."}), 404
            
        df = df.reset_index()
        # Clean MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        # Ensure we have all required columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Name'] = ticker
        
        # Save as CSV in DATA_DIR
        filename = f"{ticker}_2018-01-01_to_2026-07-15.csv"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "name": ticker,
            "rows": len(df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
