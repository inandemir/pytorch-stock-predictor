import unittest
import os
from stock_predictor import StockPredictorEngine

class TestStockPredictor(unittest.TestCase):
    def setUp(self):
        self.data_dir = r"c:\Users\inan demir\Desktop\microsoft projem\local_rag_app\data"
        self.predictor = StockPredictorEngine(data_dir=self.data_dir)

    def test_list_stocks(self):
        stocks = self.predictor.list_stocks()
        self.assertTrue(len(stocks) > 0)
        self.assertTrue(any(s["name"] == "AMZN" for s in stocks))

    def test_train_and_predict(self):
        # We will train for only 2 epochs to make it run fast
        stock_file = "AMZN_2006-01-01_to_2018-01-01.csv"
        print(f"\nTesting training on {stock_file} for 2 epochs...")
        
        metrics = self.predictor.train(stock_file, epochs=2)
        
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("weights_path", metrics)
        self.assertTrue(os.path.exists(metrics["weights_path"]))
        
        # Test chart saving
        chart_path = r"c:\Users\inan demir\Desktop\microsoft projem\local_rag_app\static\stock_chart.png"
        self.assertTrue(os.path.exists(chart_path))
        
        # Test prediction
        last_prices = metrics["last_prices"]
        self.assertEqual(len(last_prices), 20)
        
        prediction = self.predictor.predict_next(last_prices)
        print(f"Test prediction: {prediction:.2f}")
        self.assertTrue(prediction > 0)

if __name__ == "__main__":
    unittest.main()
