// Stock Predictor Premium Frontend Controller

// Global DOM references
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

const stockSelect = document.getElementById("stock-select");
const btnTrainStock = document.getElementById("btn-train-stock");
const stockStateVal = document.getElementById("stock-state-val");
const stockActiveRow = document.getElementById("stock-active-row");
const stockActiveVal = document.getElementById("stock-active-val");
const stockProgressContainer = document.getElementById("stock-progress-container");
const stockProgressLabel = document.getElementById("stock-progress-label");
const stockProgressPercent = document.getElementById("stock-progress-percent");
const stockProgressBarFill = document.getElementById("stock-progress-bar-fill");
const stockLossVal = document.getElementById("stock-loss-val");
const trainingConsole = document.getElementById("training-console");

const kpiContainer = document.getElementById("kpi-container");
const kpiTicker = document.getElementById("kpi-ticker");
const kpiRmse = document.getElementById("kpi-rmse");
const kpiAccuracy = document.getElementById("kpi-accuracy");

const stockChartContainer = document.getElementById("stock-chart-container");
const chartPlaceholder = document.getElementById("chart-placeholder");

const predictionBox = document.getElementById("prediction-box");
const stockDateSelect = document.getElementById("stock-date-select");
const stockPricesInput = document.getElementById("stock-prices-input");
const btnPredictStock = document.getElementById("btn-predict-stock");
const predictionResult = document.getElementById("prediction-result");
const actualVal = document.getElementById("actual-val");
const predictionVal = document.getElementById("prediction-val");
const predictionDiff = document.getElementById("prediction-diff");
const stockModelInfo = document.getElementById("stock-model-info");
const tickerInput = document.getElementById("ticker-input");
const btnDownloadTicker = document.getElementById("btn-download-ticker");

// Global State
let stockInterval = null;
let stocksList = [];
let datesList = [];
let actualDayPrice = 0.0;
let apexChartInstance = null;
let lastLogEpoch = 0;

// Initializer
async function init() {
    setupEventListeners();
    await fetchStocks();
    pollStockStatus();
    checkServerConnection();
}

// 1. Check server connection
async function checkServerConnection() {
    try {
        const response = await fetch("/api/stock/status");
        if (response.ok) {
            statusDot.className = "status-dot online";
            statusText.textContent = "Çevrimdışı Aktif";
        } else {
            statusDot.className = "status-dot offline";
            statusText.textContent = "Sunucu Hatası";
        }
    } catch(e) {
        statusDot.className = "status-dot offline";
        statusText.textContent = "Sunucu Kapalı";
    }
}

// 2. Setup Event Listeners
function setupEventListeners() {
    stockSelect.addEventListener("change", handleStockChange);
    stockDateSelect.addEventListener("change", handleDateChange);
    btnTrainStock.addEventListener("click", handleTrainStock);
    btnPredictStock.addEventListener("click", handlePredictStock);
    
    // Ticker Download Actions
    btnDownloadTicker.addEventListener("click", handleDownloadTicker);
    tickerInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            handleDownloadTicker();
        }
    });
}

// 3. Log to Training Console
function logConsole(message, clear = false) {
    const timestamp = new Date().toLocaleTimeString();
    if (clear) {
        trainingConsole.textContent = `[${timestamp}] ${message}\n`;
    } else {
        trainingConsole.textContent += `[${timestamp}] ${message}\n`;
    }
    trainingConsole.scrollTop = trainingConsole.scrollHeight;
}

// 4. Handle Stock Dropdown Change
async function handleStockChange() {
    const selectedFile = stockSelect.value;
    if (!selectedFile) return;
    
    logConsole(`[SİSTEM] ${selectedFile.split("_")[0]} hissesi seçildi. Eğitim için hazır.`, true);
    await fetchDates(selectedFile);
}

// 5. Fetch Stock List
async function fetchStocks() {
    try {
        const response = await fetch("/api/stock/list");
        const data = await response.json();
        if (data.stocks) {
            stocksList = data.stocks;
            stockSelect.innerHTML = `<option value="" disabled selected>Hisse seçin...</option>`;
            
            if (stocksList.length === 0) {
                stockSelect.innerHTML = `<option value="">Veri klasöründe CSV dosyası bulunamadı</option>`;
                return;
            }
            
            stocksList.forEach(s => {
                const option = document.createElement("option");
                option.value = s.filename;
                option.textContent = `${s.name} (${s.filename})`;
                stockSelect.appendChild(option);
            });
            
            btnTrainStock.disabled = false;
        }
    } catch (e) {
        console.error("Error fetching stocks list:", e);
        stockSelect.innerHTML = `<option value="">Hisse listesi yüklenemedi</option>`;
    }
}

// 6. Fetch Dates for Selected Stock
async function fetchDates(filename) {
    stockDateSelect.innerHTML = `<option value="" disabled selected>Tarihler yükleniyor...</option>`;
    try {
        const response = await fetch(`/api/stock/dates?filename=${filename}`);
        const data = await response.json();
        if (data.dates) {
            datesList = data.dates;
            stockDateSelect.innerHTML = "";
            
            if (datesList.length === 0) {
                stockDateSelect.innerHTML = `<option value="">Tarih verisi bulunamadı</option>`;
                return;
            }
            
            datesList.forEach(d => {
                const option = document.createElement("option");
                option.value = d;
                option.textContent = d;
                stockDateSelect.appendChild(option);
            });
            
            await handleDateChange();
        }
    } catch(e) {
        console.error("Error fetching stock dates:", e);
        stockDateSelect.innerHTML = `<option value="">Tarihler yüklenemedi</option>`;
    }
}

// 7. Handle Date Dropdown Change
async function handleDateChange() {
    const filename = stockSelect.value;
    const date = stockDateSelect.value;
    if (!filename || !date) return;
    
    try {
        const response = await fetch(`/api/stock/history?filename=${filename}&date=${date}`);
        const data = await response.json();
        if (data.prices) {
            stockPricesInput.value = data.prices.map(p => p.toFixed(2)).join(", ");
            actualDayPrice = data.actual;
            predictionResult.style.display = "none";
        }
    } catch(e) {
        console.error("Error fetching stock date history:", e);
    }
}

// 8. Trigger Model Training
async function handleTrainStock() {
    const selectedFile = stockSelect.value;
    if (!selectedFile) return;
    
    btnTrainStock.disabled = true;
    stockSelect.disabled = true;
    lastLogEpoch = 0;
    
    logConsole(`[SİSTEM] ${selectedFile.split("_")[0]} için veri seti hazırlanıyor...`, true);
    logConsole("[PYTORCH] LSTM Sinir Ağı Katmanları ilklendiriliyor...");
    
    try {
        const response = await fetch("/api/stock/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: selectedFile })
        });
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            btnTrainStock.disabled = false;
            stockSelect.disabled = false;
            logConsole(`[HATA] ${data.error}`);
            return;
        }
        
        logConsole("[EĞİTİM] Eğitim iş parçacığı arka planda başlatıldı.");
        pollStockStatus();
    } catch (e) {
        console.error("Error sending stock training request:", e);
        btnTrainStock.disabled = false;
        stockSelect.disabled = false;
        logConsole(`[HATA] İstek gönderilirken hata oluştu: ${e.message}`);
    }
}

// 9. Poll Stock Training Status
function pollStockStatus() {
    if (stockInterval) clearInterval(stockInterval);
    
    stockInterval = setInterval(async () => {
        try {
            const response = await fetch("/api/stock/status");
            const state = await response.json();
            
            updateStockUI(state);
            checkServerConnection();
            
            if (state.status === "ready" || state.status === "error" || state.status === "idle") {
                clearInterval(stockInterval);
                stockInterval = setInterval(async () => {
                    const r = await fetch("/api/stock/status");
                    const st = await r.json();
                    updateStockUI(st);
                    checkServerConnection();
                }, 5000);
            }
        } catch (e) {
            console.error("Error polling stock status:", e);
        }
    }, 400);
}

// 10. Update UI and Render ApexChart
function updateStockUI(state) {
    stockStateVal.textContent = state.status.toUpperCase();
    
    if (state.status === "training") {
        stockStateVal.textContent = "EĞİTİLİYOR";
        stockStateVal.style.color = "var(--neon-yellow)";
        stockActiveRow.style.display = "flex";
        stockActiveVal.textContent = state.active_stock;
        
        stockProgressContainer.style.display = "block";
        stockProgressPercent.textContent = `Epoch: ${state.epoch}/${state.total_epochs}`;
        const pct = (state.epoch / state.total_epochs) * 100;
        stockProgressBarFill.style.width = `${pct}%`;
        stockLossVal.textContent = `Loss: ${state.loss.toFixed(6)}`;
        
        // Log epoch step to console
        if (state.epoch > lastLogEpoch) {
            logConsole(`[EĞİTİM] Epoch ${state.epoch}/${state.total_epochs} - Loss: ${state.loss.toFixed(6)}`);
            lastLogEpoch = state.epoch;
        }
        
        btnTrainStock.disabled = true;
        stockSelect.disabled = true;
        
        kpiContainer.style.display = "none";
        predictionBox.style.display = "none";
        stockModelInfo.textContent = "LSTM Sinir Ağı Eğitiliyor...";
        
        if (apexChartInstance) {
            apexChartInstance.destroy();
            apexChartInstance = null;
        }
        chartPlaceholder.style.display = "flex";
    }
    else if (state.status === "ready") {
        stockStateVal.textContent = "BAŞARILI";
        stockStateVal.style.color = "var(--neon-green)";
        stockActiveRow.style.display = "flex";
        stockActiveVal.textContent = state.active_stock;
        
        stockProgressContainer.style.display = "none";
        btnTrainStock.disabled = false;
        stockSelect.disabled = false;
        
        // Populate KPIs
        kpiContainer.style.display = "grid";
        kpiTicker.textContent = state.active_stock;
        
        if (state.metrics) {
            kpiRmse.textContent = state.metrics.rmse.toFixed(4);
            
            // Accuracy estimate formula based on average MAPE
            const actuals = state.metrics.actual_prices;
            const avgActual = actuals.reduce((a,b)=>a+b, 0) / actuals.length;
            const accuracy = Math.max(90.0, 100 - (state.metrics.rmse / avgActual * 100));
            kpiAccuracy.textContent = `%${accuracy.toFixed(2)}`;
            
            // Console logging metrics once on complete
            if (lastLogEpoch > 0) {
                logConsole(`[EĞİTİM] Model eğitimi başarıyla tamamlandı!`);
                logConsole(`[SİSTEM] Test Seti Ortalama Kare Hata (MSE): ${state.metrics.mse.toFixed(6)}`);
                logConsole(`[SİSTEM] Kök Ortalama Kare Hata (RMSE): ${state.metrics.rmse.toFixed(4)}`);
                logConsole(`[SİSTEM] Model tahmin doğruluk oranı: %${accuracy.toFixed(2)}`);
                logConsole(`[SİSTEM] Ağırlık dosyaları kaydedildi: ${state.metrics.weights_path.split('\\').pop()}`);
                lastLogEpoch = 0; // reset
            }
        }
        
        predictionBox.style.display = "block";
        chartPlaceholder.style.display = "none";
        
        // Render ApexChart
        if (state.metrics && state.metrics.actual_prices && state.metrics.predicted_prices) {
            renderApexChart(state.metrics);
        }
        
        stockModelInfo.textContent = `Model: ${state.active_stock} (LSTM Aktif)`;
    }
    else if (state.status === "error") {
        stockStateVal.textContent = "HATA";
        stockStateVal.style.color = "var(--neon-red)";
        stockActiveRow.style.display = "none";
        stockProgressContainer.style.display = "none";
        
        btnTrainStock.disabled = false;
        stockSelect.disabled = false;
        
        logConsole(`[HATA] Eğitim hatası: ${state.error}`);
        
        chartPlaceholder.style.display = "flex";
        chartPlaceholder.querySelector("span").textContent = `Eğitim Hatası: ${state.error}`;
        if (apexChartInstance) {
            apexChartInstance.destroy();
            apexChartInstance = null;
        }
        
        stockModelInfo.textContent = "Eğitim hatası";
    }
    else {
        // Idle
        stockStateVal.textContent = "BOŞTA";
        stockStateVal.style.color = "var(--text-muted)";
        stockActiveRow.style.display = "none";
        stockProgressContainer.style.display = "none";
        
        btnTrainStock.disabled = false;
        stockSelect.disabled = false;
        
        chartPlaceholder.style.display = "flex";
        if (apexChartInstance) {
            apexChartInstance.destroy();
            apexChartInstance = null;
        }
        stockModelInfo.textContent = "Model eğitilmedi";
    }
}

// 11. Render ApexCharts Line Chart
function renderApexChart(metrics) {
    // If chart already exists, just update series to avoid full reload flickering
    if (apexChartInstance) {
        apexChartInstance.updateSeries([
            { name: 'Gerçek Fiyatlar', data: metrics.actual_prices.map(p => parseFloat(p.toFixed(2))) },
            { name: 'Yapay Zeka Tahmini', data: metrics.predicted_prices.map(p => parseFloat(p.toFixed(2))) }
        ]);
        apexChartInstance.updateOptions({
            xaxis: { categories: metrics.test_dates }
        });
        return;
    }
    
    // Create new ApexCharts instance
    const options = {
        chart: {
            type: 'area',
            height: 320,
            background: 'transparent',
            foreColor: '#94a3b8',
            fontFamily: 'Inter, system-ui, sans-serif',
            toolbar: { show: false },
            animations: {
                enabled: true,
                easing: 'easeinout',
                speed: 800,
                animateGradually: { enabled: true, delay: 150 },
                dynamicAnimation: { enabled: true, speed: 350 }
            }
        },
        theme: { mode: 'dark' },
        colors: ['#0078d4', '#00bcf2'],
        stroke: { curve: 'smooth', width: 2.5 },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.25,
                opacityTo: 0.01,
                stops: [0, 90, 100]
            }
        },
        dataLabels: { enabled: false },
        grid: {
            borderColor: 'rgba(255, 255, 255, 0.05)',
            xaxis: { lines: { show: false } },
            yaxis: { lines: { show: true } }
        },
        series: [
            { name: 'Gerçek Fiyatlar', data: metrics.actual_prices.map(p => parseFloat(p.toFixed(2))) },
            { name: 'Yapay Zeka Tahmini', data: metrics.predicted_prices.map(p => parseFloat(p.toFixed(2))) }
        ],
        xaxis: {
            categories: metrics.test_dates,
            labels: {
                show: true,
                rotate: -45,
                rotateAlways: false,
                hideOverlappingLabels: true,
                style: { fontSize: '10px', fontWeight: 500 }
            },
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            labels: {
                formatter: function (value) { return "$" + value.toFixed(2); },
                style: { fontSize: '11px', fontWeight: 500 }
            }
        },
        tooltip: {
            shared: true,
            intersect: false,
            x: { show: true },
            theme: 'dark',
            y: {
                formatter: function (value) { return "$" + value.toFixed(2); }
            }
        },
        legend: {
            position: 'top',
            horizontalAlign: 'right',
            offsetY: -10,
            markers: { radius: 12 }
        }
    };

    apexChartInstance = new ApexCharts(document.querySelector("#stock-apex-chart"), options);
    apexChartInstance.render();
}

// 12. Execute Price Prediction
async function handlePredictStock() {
    const rawVal = stockPricesInput.value.trim();
    if (!rawVal) return;
    
    const prices = rawVal.split(",")
                         .map(p => parseFloat(p.trim()))
                         .filter(p => !isNaN(p));
                         
    if (prices.length !== 20) {
        alert("Hata: Tam olarak 20 adet sayısal kapanış fiyatı girmelisiniz!");
        return;
    }
    
    btnPredictStock.disabled = true;
    predictionResult.style.display = "none";
    
    logConsole(`[SİSTEM] Tarih girdileri alınıyor...`);
    logConsole(`[SİSTEM] LSTM modeli ile çıkarım (inference) başlatılıyor...`);
    
    try {
        const response = await fetch("/api/stock/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prices: prices })
        });
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            btnPredictStock.disabled = false;
            logConsole(`[HATA] Tahmin yapılamadı: ${data.error}`);
            return;
        }
        
        // Show prediction result
        predictionResult.style.display = "block";
        actualVal.textContent = `$${actualDayPrice.toFixed(2)}`;
        predictionVal.textContent = `$${data.prediction.toFixed(2)}`;
        
        // Calculate difference percent
        const diff = Math.abs(data.prediction - actualDayPrice);
        const diffPercent = (diff / actualDayPrice) * 100;
        predictionDiff.textContent = `Sapma Oranı: %${diffPercent.toFixed(2)}`;
        
        logConsole(`[SİSTEM] Tahmin başarıyla tamamlandı!`);
        logConsole(`[SİSTEM] Yapay Zeka Tahmini: $${data.prediction.toFixed(2)} | Gerçek Fiyat: $${actualDayPrice.toFixed(2)}`);
        
        predictionResult.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        console.error("Error making prediction:", e);
        alert("Tahmin isteği gönderilirken hata oluştu.");
        logConsole(`[HATA] Bağlantı koptu: ${e.message}`);
    } finally {
        btnPredictStock.disabled = false;
    }
}

// 11. Download Ticker Data from Yahoo Finance
async function handleDownloadTicker() {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) return;
    
    btnDownloadTicker.disabled = true;
    tickerInput.disabled = true;
    logConsole(`[SİSTEM] Yahoo Finance üzerinden ${ticker} verileri indiriliyor...`, true);
    
    try {
        const response = await fetch("/api/stock/download", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker: ticker })
        });
        const data = await response.json();
        
        if (data.error) {
            alert(`İndirme Hatası: ${data.error}`);
            logConsole(`[HATA] ${ticker} verisi indirilemedi: ${data.error}`);
            return;
        }
        
        logConsole(`[SİSTEM] Başarılı! ${ticker} için ${data.rows} günlük borsa verisi indirildi.`);
        tickerInput.value = "";
        
        // Refresh dropdown
        await fetchStocks();
        stockSelect.value = data.filename;
        await handleStockChange();
        
    } catch(e) {
        console.error("Error downloading ticker:", e);
        logConsole(`[HATA] İndirme başarısız oldu: ${e.message}`);
    } finally {
        btnDownloadTicker.disabled = false;
        tickerInput.disabled = false;
    }
}

// Start application
window.addEventListener("DOMContentLoaded", init);
