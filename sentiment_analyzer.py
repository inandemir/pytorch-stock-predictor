import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "sentiment_vectorizer.pkl")

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        
    def train_or_load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                with open(VECTORIZER_PATH, "rb") as f:
                    self.vectorizer = pickle.load(f)
                self.is_loaded = True
                print("Loaded sentiment model from cache.")
                return
            except Exception as e:
                print(f"Error loading cached sentiment model: {e}. Retraining...")

        # Find data.csv in the parent directory
        dataset_path = os.path.join(os.path.dirname(BASE_DIR), "data.csv")
        if not os.path.exists(dataset_path):
            # Try within data folder
            dataset_path = os.path.join(BASE_DIR, "data", "data.csv")
            if not os.path.exists(dataset_path):
                print(f"Warning: Sentiment dataset data.csv not found at {dataset_path}!")
                return

        try:
            print(f"Training sentiment model on {dataset_path}...")
            df = pd.read_csv(dataset_path)
            
            # Map sentiment labels
            label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            df['label'] = df['Sentiment'].map(label_map)
            df = df.dropna(subset=['Sentence', 'label'])
            
            # Vectorize sentences
            self.vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
            X = self.vectorizer.fit_transform(df['Sentence'])
            y = df['label'].values
            
            # Train Logistic Regression
            self.model = LogisticRegression(C=1.0, max_iter=200)
            self.model.fit(X, y)
            
            # Cache model
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.model, f)
            with open(VECTORIZER_PATH, "wb") as f:
                pickle.dump(self.vectorizer, f)
                
            self.is_loaded = True
            print("Sentiment model trained and cached successfully.")
        except Exception as e:
            print(f"Error training sentiment model: {e}")

    def analyze(self, text):
        if not self.is_loaded or self.model is None or self.vectorizer is None:
            return 0.0
            
        try:
            vec = self.vectorizer.transform([text])
            score = self.model.predict_proba(vec)[0]
            # score[0] corresponds to negative (-1.0), score[1] to neutral (0.0), score[2] to positive (1.0)
            sentiment_value = (-1.0 * score[0]) + (0.0 * score[1]) + (1.0 * score[2])
            return float(sentiment_value)
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return 0.0
