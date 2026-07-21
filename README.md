# 📈 Local RAG Finansal Borsa Tahmin & Raporlama Ajanı

Bu proje; yerel, gizlilik odaklı ve tamamen **offline (çevrimdışı) çalışabilen** bir finansal borsa analizi, LSTM fiyat tahmini, duygu analizi ve akıllı RAG (Retrieval-Augmented Generation) raporlama platformudur.

Platform, derin öğrenme temelli fiyat öngörüleri ile piyasa duygu iklimini ve özel analiz notlarınızı harmanlayarak kişiselleştirilmiş yatırımcı raporları oluşturur.

---

## 🚀 Öne Çıkan Özellikler

1.  **PyTorch LSTM Fiyat Öngörüsü:**
    *   Hisse senedi verilerini Yahoo Finance üzerinden otomatik indirir.
    *   Geçmiş fiyatları kullanarak yerel bir **LSTM** modeli eğitir ve sonraki günün yönünü ve fiyatını tahmin eder.
2.  **Haber NLP Duygu Analizi Sınıflandırıcısı:**
    *   Yahoo Finance haberlerini toplar, makine öğrenmesi sınıflandırıcısıyla analiz ederek, hissenin piyasadaki olumlu/olumsuz hava durumunu ölçer ve görselleştirir.
3.  **Kişisel Yatırımcı Profili & Akıllı RAG Raporlama:**
    *   İnternete bağlı olmadan, dökümanlardan semantik arama (RAG) yaparak kişiye özel profesyonel bir borsa raporu yazar.
    *   Raporlama ve sohbet yanıtlarını belirlediğiniz **Risk Toleransı** (Korumacı, Dengeli, Agresif) ve **Yatırım Vadesi** (Kısa, Orta, Uzun) tercihlerinize göre şekillendirir.
4.  **Dinamik Renk Ortamları / Temalar:**
    *   4 farklı neon tema seçeneği (Neon Mavi, Neon Mor, Zümrüt Yeşili, Kızıl Şafak).
5.  **Dinamik Grafik Para Birimi Entegrasyonu:**
    *   Seçtiğiniz para birimine göre ($, ₺, €) eksen değerleri ve grafik tooltipleri anlık güncellenir.
6.  **Sohbet Geçmişini Sıfırlama:**
    *   Tek tıkla sohbet geçmişini temizleme butonu.

---

## 🛠️ Kullanılan Teknolojiler

*   **Backend:** Flask (Python), Flask-CORS, `yfinance`, `pandas`.
*   **Derin Öğrenme / Makine Öğrenmesi:** PyTorch (LSTM), Scikit-Learn (Logistic Regression & TfidfVectorizer).
*   **Yerel Dil Modeli (LLM):** Foundry Local SDK (`qwen2.5-0.5b`).
*   **Frontend:** Vanilla HTML5, CSS3 (Glassmorphism Premium Tema), Vanilla Javascript.
*   **Grafikler:** ApexCharts.

---

## 📦 Kurulum ve Çalıştırma

### 1. Gereksinimlerin Yüklenmesi
Gerekli Python paketlerini yüklemek için terminalde şu komutu çalıştırın:
```powershell
pip install -r requirements.txt
```

### 2. Uygulamanın Başlatılması
Yerel sunucuyu ayağa kaldırmak için:
```powershell
python app.py
```
Sunucu çalıştıktan sonra tarayıcınızda **[http://127.0.0.1:5000](http://127.0.0.1:5000)** adresine giderek uygulamayı kullanmaya başlayabilirsiniz.

---

## 👥 Geliştirici
*   **İnan Demir** - [GitHub Profiliniz](https://github.com/inandemir)
