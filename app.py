import os
import threading
import json
import pandas as pd
import yfinance as yf
import openai
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from stock_predictor import StockPredictorEngine
from foundry_local_sdk import Configuration, FoundryLocalManager

app = Flask(__name__, static_folder="static")
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

def is_safe_filename(filename):
    if not filename:
        return False
    if ".." in filename or "/" in filename or "\\" in filename:
        return False
    if not filename.endswith(".csv"):
        return False
    return True

# Initialize Stock Predictor
stock_predictor = StockPredictorEngine(data_dir=DATA_DIR)

from sentiment_analyzer import SentimentAnalyzer
sentiment_analyzer = SentimentAnalyzer()
thread = threading.Thread(target=sentiment_analyzer.train_or_load)
thread.daemon = True
thread.start()

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

# Global LLM (RAG) State
llm_state = {
    "status": "idle", # idle, loading, ready, error
    "progress": 0,
    "error": None
}

foundry_manager = None
openai_client = None

def init_foundry():
    global foundry_manager
    try:
        config = Configuration(app_name="local-rag-assistant")
        FoundryLocalManager.initialize(config)
        foundry_manager = FoundryLocalManager.instance
        print("Foundry Local Manager initialized.")
    except Exception as e:
        print(f"Error initializing Foundry Local: {e}")

init_foundry()

def bg_load_llm():
    global llm_state, foundry_manager, openai_client
    try:
        llm_state["status"] = "loading"
        llm_state["progress"] = 20
        llm_state["error"] = None
        
        print("Loading local LLM model qwen2.5-0.5b...")
        model_info = foundry_manager.catalog.get_model("qwen2.5-0.5b")
        llm_state["progress"] = 50
        
        model_info.load()
        llm_state["progress"] = 75
        
        if not foundry_manager.urls:
            foundry_manager.start_web_service()
            
        openai_client = openai.OpenAI(
            base_url=f"{foundry_manager.urls[0]}/v1",
            api_key="local"
        )
        
        llm_state["progress"] = 100
        llm_state["status"] = "ready"
        print("Local LLM model qwen2.5-0.5b is ready!")
    except Exception as e:
        llm_state["status"] = "error"
        llm_state["error"] = str(e)
        print(f"Error loading local LLM: {e}")

def bg_train_stock(stock_filename, epochs=25):
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
            
        metrics = stock_predictor.train(stock_filename, progress_callback=progress_callback, epochs=epochs)
        
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
    epochs = data.get("epochs", 25)
    try:
        epochs = int(epochs)
    except:
        epochs = 25
        
    if not is_safe_filename(stock_filename):
        return jsonify({"error": "Geçersiz veya güvensiz dosya adı!"}), 400
        
    if stock_training_state["status"] == "training":
        return jsonify({"error": "A model is already training"}), 400
        
    thread = threading.Thread(target=bg_train_stock, args=(stock_filename, epochs))
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
    if not is_safe_filename(filename):
        return jsonify({"error": "Geçersiz veya güvensiz dosya adı!"}), 400
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
    if not is_safe_filename(filename) or not date:
        return jsonify({"error": "Geçersiz veya güvensiz dosya adı ya da tarih parametresi eksik!"}), 400
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
    # Validate ticker input: only alphanumeric, dots and hyphens
    if not ticker or not all(c.isalnum() or c in "-." for c in ticker):
        return jsonify({"error": "Geçersiz veya güvensiz borsa sembolü!"}), 400
        
    try:
        df = yf.download(ticker, start="2018-01-01", end="2026-07-15")
        if df.empty:
            return jsonify({"error": f"{ticker} sembolü için veri bulunamadı."}), 404
            
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Name'] = ticker
        
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

# 8. Get LLM Load Status
@app.route("/api/llm/status", methods=["GET"])
def get_llm_status():
    return jsonify(llm_state)

# 9. Trigger LLM Load
@app.route("/api/llm/load", methods=["POST"])
def load_llm():
    if llm_state["status"] not in ["loading", "ready"]:
        thread = threading.Thread(target=bg_load_llm)
        thread.daemon = True
        thread.start()
    return jsonify(llm_state)

# 10. Generate RAG Financial Report (SSE Streaming)
@app.route("/api/stock/report", methods=["GET"])
def generate_stock_report():
    filename = request.args.get("filename")
    temp = request.args.get("temperature", 0.3)
    max_tok = request.args.get("max_tokens", 600)
    freq_pen = request.args.get("frequency_penalty", 1.2)
    risk = request.args.get("risk", "medium")
    horizon = request.args.get("horizon", "medium")
    currency = request.args.get("currency", "USD")
    
    try:
        temp = float(temp)
        max_tok = int(max_tok)
        freq_pen = float(freq_pen)
    except:
        temp = 0.3
        max_tok = 600
        freq_pen = 1.2
        
    if not is_safe_filename(filename):
        return jsonify({"error": "Geçersiz veya güvensiz dosya adı!"}), 400
        
    def sse_generator():
        global llm_state, openai_client
        
        # 1. Check if LLM is ready. If not, trigger load and wait.
        if llm_state["status"] != "ready":
            yield f"data: {json.dumps({'type': 'status', 'text': 'Yapay zeka modeli belleğe yükleniyor (qwen2.5-0.5b)...'})}\n\n"
            if llm_state["status"] not in ["loading"]:
                bg_load_llm()
                
            import time
            while llm_state["status"] == "loading":
                prog = llm_state["progress"]
                yield f"data: {json.dumps({'type': 'status', 'text': f'Model yükleniyor... %{prog}'})}\n\n"
                time.sleep(0.5)
                
            if llm_state["status"] == "error":
                err_msg = llm_state["error"]
                yield f"data: {json.dumps({'type': 'error', 'text': f'Model yüklenemedi: {err_msg}'})}\n\n"
                return
                
        # 2. Get stock price data & LSTM prediction
        ticker_symbol = filename.split("_")[0]
        yield f"data: {json.dumps({'type': 'status', 'text': f'LSTM model tahmini hesaplanıyor ve {ticker_symbol} son verileri alınıyor...'})}\n\n"
        
        try:
            # Load last 20 close prices from file
            filepath = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(filepath)
            df = df.sort_values('Date')
            last_20_prices = df['Close'].values[-20:].tolist()
            last_close = last_20_prices[-1]
            
            # Predict
            pred_price = stock_predictor.predict_next(last_20_prices)
            price_change = pred_price - last_close
            pct_change = (price_change / last_close) * 100
            trend_direction = "YUKARI (Artış)" if price_change >= 0 else "AŞAĞI (Azalış)"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'LSTM tahmin hatası: {str(e)}'})}\n\n"
            return
            
        # 3. Retrieve Latest news from yfinance (RAG Retrieval)
        yield f"data: {json.dumps({'type': 'status', 'text': f'Yahoo Finance üzerinden {ticker_symbol} hakkında en güncel haberler toplanıyor (RAG)...'})}\n\n"
        
        retrieved_news = []
        try:
            ticker = yf.Ticker(ticker_symbol)
            news_items = ticker.news[:5]
            for item in news_items:
                content = item.get('content', {})
                title = content.get('title') or item.get('title')
                summary = content.get('summary') or content.get('description') or item.get('summary') or item.get('description') or ""
                pub_date = content.get('pubDate') or item.get('pubDate') or ""
                provider = content.get('provider', {}).get('displayName', item.get('provider', {}).get('displayName', 'Bilinmeyen'))
                if title:
                    # Run Sentiment Analysis
                    sentiment_score = sentiment_analyzer.analyze(title + " " + summary)
                    retrieved_news.append({
                        "title": title,
                        "summary": summary[:250] + "..." if len(summary) > 250 else summary,
                        "pub_date": pub_date,
                        "provider": provider,
                        "sentiment": sentiment_score
                    })
        except Exception as e:
            yield f"data: {json.dumps({'type': 'status', 'text': f'Haberler çekilirken hata oluştu fakat analize devam ediliyor: {str(e)}'})}\n\n"
            
        # 4. Synthesize prompt (Augmentation)
        yield f"data: {json.dumps({'type': 'status', 'text': 'Veriler sentezleniyor ve Yapay Zeka Analizi yazılıyor...'})}\n\n"
        
        news_context = ""
        avg_sentiment = 0.0
        if retrieved_news:
            total_sentiment = 0.0
            for idx, n in enumerate(retrieved_news):
                sent_label = "Nötr"
                if n['sentiment'] > 0.15:
                    sent_label = f"Olumlu (Skor: {n['sentiment']:+.2f})"
                elif n['sentiment'] < -0.15:
                    sent_label = f"Olumsuz (Skor: {n['sentiment']:+.2f})"
                else:
                    sent_label = f"Nötr (Skor: {n['sentiment']:+.2f})"
                    
                news_context += f"{idx+1}. Başlık: {n['title']}\n   Kaynak: {n['provider']} ({n['pub_date']})\n   Haber Analiz Duygusu: {sent_label}\n   Özet: {n['summary']}\n\n"
                total_sentiment += n['sentiment']
            avg_sentiment = total_sentiment / len(retrieved_news)
        else:
            news_context = "Son haberlere ulaşılamadı.\n"
            
        # Risk profile explanations
        risk_labels = {"low": "Korumacı (Düşük Risk toleranslı, güvenli enstrümanlar odaklı)", 
                       "medium": "Dengeli (Orta Risk toleranslı)", 
                       "high": "Agresif (Yüksek Risk toleranslı, volatiliteyi fırsat gören)"}
        horizon_labels = {"short": "Kısa Vade (Haftalık/Aylık kazanç odaklı)", 
                          "medium": "Orta Vade (Aylık/Yıllık)", 
                          "long": "Uzun Vade (Yıllık yatırım ve kalıcı büyüme odaklı)"}
        
        prompt = f"""Sen yerel ve çevrimdışı çalışan, uzman bir Yapay Zeka Finansal Analistisin. 
Aşağıdaki LSTM fiyat tahmini verilerini ve hisseye dair en son haber başlıklarını/özetlerini harmanlayarak yatırımcılara yönelik Türkçe detaylı bir borsa analiz raporu yaz.

Kullanıcı Yatırım Profili & Tercihleri:
- Risk Toleransı: {risk_labels.get(risk, risk)}
- Yatırım Vadesi: {horizon_labels.get(horizon, horizon)}
- Tercih Edilen Para Birimi: {currency}

Hisse Senedi Ticker: {ticker_symbol}
Yapay Zeka (LSTM) Tahmin Verileri:
- Son Kapanış Fiyatı: {currency} {last_close:.2f}
- Yarın İçin LSTM Tahmini: {currency} {pred_price:.2f}
- Tahmin Edilen Değişim: {price_change:+.2f} ({pct_change:+.2f}%)
- Öngörülen Trend: {trend_direction}

Yahoo Finance'ten Alınan Son Haberler ve NLP Duygu Analizi Skorları (RAG Bağlamı):
- Güncel Haberlerin Ortalama Duygu Skoru: {avg_sentiment:+.2f} (Ölçek: -1.0 olumsuz, 0.0 nötr, +1.0 olumlu)

Haber Listesi:
{news_context}

Lütfen bu rapora ve yatırım tavsiyelerine kullanıcının bu risk toleransına ve vade profiline göre şekil ver. Örneğin risk toleransı yüksekse daha agresif fırsatları, düşükse güvenli limanları ve riskleri öne çıkar.

Lütfen raporu tam olarak şu Markdown başlıkları ve yapısıyla yaz:

### 📈 LSTM Fiyat Analizi
(Tahmini fiyatı, son kapanış fiyatını ve yönünü detaylandırarak açıklayın. LSTM modelimizin öngörüsünü yorumlayın.)

### 📰 Haber Başlıkları & Gelişmeler
(RAG ile çekilen haber başlıklarını inceleyerek, olumlu ve olumsuz haberleri listele ve bu haberlerin hisseye olası etkilerini finansal açıdan yorumla.)

### 💡 AI Yatırım Tavsiyesi & Değerlendirme
(LSTM sayısal tahmini ile haberlerin analizini harmanlayarak, yerel LLM olarak bu hisse senedi hakkında kullanıcının risk toleransına ve vadesine uygun bir Türkçe yatırım değerlendirmesi yaz. Yatırımcılara somut öneriler sun.)
"""
        
        # 5. Call LLM (Generation)
        try:
            stream = openai_client.chat.completions.create(
                model="qwen2.5-0.5b",
                messages=[
                    {"role": "system", "content": "Sen profesyonel bir finans analisti ve borsa danışmanısın. Türkçe konuşuyorsun. Sayısal verileri ve haber özetlerini çok iyi analiz edersin."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=max_tok,
                frequency_penalty=freq_pen,
                presence_penalty=1.0,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    txt = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'content', 'text': txt})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'LLM Çıkarım Hatası: {str(e)}'})}\n\n"
            
    return Response(sse_generator(), mimetype="text/event-stream")

# 11. Upload Custom Stock Analysis Note / Document
@app.route("/api/stock/upload_note", methods=["POST"])
def upload_stock_note():
    data = request.json or {}
    ticker = data.get("ticker", "").upper().strip()
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    
    # Validation
    if not ticker or not all(c.isalnum() or c in "-." for c in ticker):
        return jsonify({"error": "Geçersiz veya güvensiz borsa sembolü!"}), 400
    if not title or not content:
        return jsonify({"error": "Başlık ve içerik gereklidir!"}), 400
        
    ticker_dir = os.path.join(KNOWLEDGE_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Safe filename creation from title
    safe_title = "".join([c if c.isalnum() else "_" for c in title])
    import time
    filename = f"{int(time.time())}_{safe_title[:30]}.txt"
    filepath = os.path.join(ticker_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Başlık: {title}\nİçerik: {content}\n")
        return jsonify({"success": True, "message": f"{ticker} için yeni bilgi notu kaydedildi."})
    except Exception as e:
        return jsonify({"error": f"Dosya kaydedilemedi: {str(e)}"}), 500

# 12. Interactive Financial Chat Assistant using Semantic RAG
@app.route("/api/stock/chat", methods=["GET"])
def chat_stock_agent():
    filename = request.args.get("filename")
    query = request.args.get("query", "").strip()
    temp = request.args.get("temperature", 0.3)
    max_tok = request.args.get("max_tokens", 600)
    freq_pen = request.args.get("frequency_penalty", 1.2)
    risk = request.args.get("risk", "medium")
    horizon = request.args.get("horizon", "medium")
    currency = request.args.get("currency", "USD")
    
    try:
        temp = float(temp)
        max_tok = int(max_tok)
        freq_pen = float(freq_pen)
    except:
        temp = 0.3
        max_tok = 600
        freq_pen = 1.2
        
    if not is_safe_filename(filename):
        return jsonify({"error": "Geçersiz veya güvensiz dosya adı!"}), 400
    if not query:
        return jsonify({"error": "Soru parametresi (query) eksik!"}), 400
        
    ticker_symbol = filename.split("_")[0]
    
    def sse_generator():
        global llm_state, openai_client
        
        # 1. Check if LLM is ready
        if llm_state["status"] != "ready":
            yield f"data: {json.dumps({'type': 'status', 'text': 'Yapay zeka modeli belleğe yükleniyor...'})}\n\n"
            if llm_state["status"] not in ["loading"]:
                bg_load_llm()
            import time
            while llm_state["status"] == "loading":
                prog = llm_state["progress"]
                yield f"data: {json.dumps({'type': 'status', 'text': f'Model yükleniyor... %{prog}'})}\n\n"
                time.sleep(0.5)
            if llm_state["status"] == "error":
                err_msg = llm_state["error"]
                yield f"data: {json.dumps({'type': 'error', 'text': f'Model yüklenemedi: {err_msg}'})}\n\n"
                return

        yield f"data: {json.dumps({'type': 'status', 'text': 'Semantik döküman araması yapılıyor...'})}\n\n"
        
        # 2. Retrieve Latest News
        retrieved_news = []
        try:
            ticker = yf.Ticker(ticker_symbol)
            news_items = ticker.news[:5]
            for item in news_items:
                content = item.get('content', {})
                title = content.get('title') or item.get('title')
                summary = content.get('summary') or content.get('description') or item.get('summary') or item.get('description') or ""
                if title:
                    retrieved_news.append({"title": title, "summary": summary})
        except Exception as e:
            print(f"Error fetching news for chat: {e}")
            
        # 3. Retrieve Custom Notes
        chunks = []
        for n in retrieved_news:
            chunks.append(f"Haber Başlığı: {n['title']}. Özet: {n['summary']}")
            
        ticker_dir = os.path.join(KNOWLEDGE_DIR, ticker_symbol)
        if os.path.exists(ticker_dir):
            for f_name in os.listdir(ticker_dir):
                if f_name.endswith(".txt"):
                    f_path = os.path.join(ticker_dir, f_name)
                    try:
                        with open(f_path, "r", encoding="utf-8") as f:
                            text = f.read()
                            for line in text.split("\n"):
                                clean_l = line.strip()
                                if len(clean_l) > 15:
                                    chunks.append(clean_l)
                    except Exception as e:
                        print(f"Error reading file for chat: {e}")
                        
        # 4. Perform TF-IDF Cosine Similarity Semantic Search
        retrieved_context = ""
        if chunks:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(chunks)
                query_vector = vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
                
                top_indices = similarities.argsort()[-3:][::-1]
                retrieved_context = "\n".join([f"- {chunks[i]}" for i in top_indices if similarities[i] > 0.02])
            except Exception as e:
                print(f"Error doing vector search: {e}")
                retrieved_context = "\n".join([f"- {chunks[i]}" for i in range(min(3, len(chunks)))])
        else:
            retrieved_context = "Hisse hakkında veri bulunamadı."
            
        if not retrieved_context.strip():
            retrieved_context = "Hisse hakkında özel not, bilgi veya haber bulunamadı."
            
        # 5. Build prompt
        risk_labels = {"low": "Korumacı (Düşük Risk)", "medium": "Dengeli (Orta Risk)", "high": "Agresif (Yüksek Risk)"}
        horizon_labels = {"short": "Kısa Vade (Haftalık/Aylık)", "medium": "Orta Vade (Aylık/Yıllık)", "long": "Uzun Vade (Yıllık)"}
        
        prompt = f"""Sen yerel ve çevrimdışı çalışan, uzman bir Yapay Zeka Finansal Analistisin. 
Aşağıdaki finansal bağlamı (RAG) kullanarak kullanıcının borsa/yatırım hakkındaki sorusunu Türkçe ve detaylı olarak yanıtla. 
Erişilen bağlamı, kullanıcının kişisel yatırımcı profiliyle (risk seviyesi ve yatırım vadesi) eşleştirerek yanıtına yön ver.

Kullanıcı Yatırım Profili:
- Risk Toleransı: {risk_labels.get(risk, risk)}
- Yatırım Vadesi: {horizon_labels.get(horizon, horizon)}
- Tercih Edilen Para Birimi: {currency}

Hisse Senedi: {ticker_symbol}
Kullanıcı Sorusu: {query}

Erişilen En Alakalı Bağlamlar (RAG):
{retrieved_context}

Lütfen yanıtını profesyonel, yapıcı ve objektif bir Türkçe ile yaz. Markdown formatı kullanabilirsin.
"""
        
        # 6. Stream Completion
        try:
            stream = openai_client.chat.completions.create(
                model="qwen2.5-0.5b",
                messages=[
                    {"role": "system", "content": "Sen profesyonel bir finans analisti ve borsa danışmanısın. Türkçe konuşuyorsun. Sayısal verileri ve haber özetlerini çok iyi analiz edersin."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=max_tok,
                frequency_penalty=freq_pen,
                presence_penalty=1.0,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    txt = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'content', 'text': txt})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'LLM Çıkarım Hatası: {str(e)}'})}\n\n"

    return Response(sse_generator(), mimetype="text/event-stream")

# 13. List all stock notes in the Knowledge Base
@app.route("/api/stock/list_notes", methods=["GET"])
def list_stock_notes():
    notes_list = []
    if os.path.exists(KNOWLEDGE_DIR):
        for ticker in os.listdir(KNOWLEDGE_DIR):
            ticker_dir = os.path.join(KNOWLEDGE_DIR, ticker)
            if os.path.isdir(ticker_dir):
                for f_name in os.listdir(ticker_dir):
                    if f_name.endswith(".txt"):
                        f_path = os.path.join(ticker_dir, f_name)
                        try:
                            parts = f_name.split("_")
                            timestamp = int(parts[0])
                            import datetime
                            date_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            with open(f_path, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                                title = lines[0].replace("Başlık: ", "").strip() if len(lines) > 0 else "Başlıksız"
                                content = "".join(lines[1:]).replace("İçerik: ", "").strip() if len(lines) > 1 else ""
                            notes_list.append({
                                "ticker": ticker,
                                "filename": f_name,
                                "date": date_str,
                                "title": title,
                                "content": content
                            })
                        except Exception as e:
                            print(f"Error reading note: {e}")
    return jsonify({"notes": notes_list})

# 14. Delete a specific stock note
@app.route("/api/stock/delete_note", methods=["POST"])
def delete_stock_note():
    data = request.json or {}
    ticker = data.get("ticker", "").upper().strip()
    filename = data.get("filename", "").strip()
    
    if not ticker or not filename:
        return jsonify({"error": "Ticker ve filename parametreleri zorunludur!"}), 400
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Geçersiz dosya adı!"}), 400
        
    filepath = os.path.join(KNOWLEDGE_DIR, ticker, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return jsonify({"success": True, "message": "Not başarıyla silindi."})
        except Exception as e:
            return jsonify({"error": f"Silme işlemi başarısız: {str(e)}"}), 500
    else:
        return jsonify({"error": "Dosya bulunamadı!"}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
