# ⚡ Electricity Demand Forecasting — Time Series Analysis & Multi-Model Comparison

---

## 📘 Overview

This project applies **four distinct forecasting methodologies** — **ARIMA**, **SARIMA**, **Facebook Prophet**, and **LSTM (Long Short-Term Memory)** — to analyze and predict **U.S. monthly electricity demand** from historical consumption data. The study leverages **168 monthly observations (Jan 2010 – Dec 2023)** sourced from the **U.S. Energy Information Administration (EIA)**, covering residential, commercial, industrial, and transportation sectors.

By combining classical statistical time series models with modern deep learning approaches, this analysis delivers **data-driven demand forecasts** and quantitative model comparison — valuable for energy policymakers, utility companies, and demand planning teams.

---

## 📂 Project Structure

```
Project_2/
├── Elec_sales.xlsx                         # Raw dataset (168 rows × 22 columns)
├── Electricity Demand Forecasting.ipynb    # Full analysis notebook (50 cells)
└── README.md                               # This file
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| **Source** | U.S. Energy Information Administration (EIA) — Monthly Form EIA-861M |
| **Time Frame** | January 2010 – December 2023 |
| **Observations** | 168 monthly records |
| **Features** | 22 columns |
| **File Format** | `.xlsx` (Excel) |
| **Missing Values** | 0 (no missing data in any column) |

### Column Descriptions

The dataset contains **sector-wise electricity metrics** across four sectors — **Residential**, **Commercial**, **Industrial**, and **Transportation** — plus aggregate totals:

| Column Group | Columns | Description |
|---|---|---|
| **Temporal** | `Year`, `Month` | Time identifiers combined into a datetime index |
| **Residential** | `Revenue_residential`, `Sales_residential`, `Customers_residential`, `Price_residential` | Revenue ($), sales (MWh), customer count, and avg. price (¢/kWh) for the residential sector |
| **Commercial** | `Revenue_commercial`, `Sales_commercial`, `Customers_commercial`, `Price_commercial` | Same metrics for the commercial sector |
| **Industrial** | `Revenue_industrial`, `Sales_industrial`, `Customers_industrial`, `Price_industrial` | Same metrics for the industrial sector |
| **Transportation** | `Revenue_transportation`, `Sales_transportation`, `Customers_transportation`, `Price_transportation` | Same metrics for the transportation sector |
| **Aggregated** | `Revenue_total`, `Sales_total`, `Customers_total`, `Price_total` | Cross-sector totals |

### Target Variable — `Sales_total`

The primary forecasting target is **`Sales_total`** — the aggregated monthly electricity consumption across all sectors.

| Statistic | Value |
|---|---|
| Count | 168 |
| Mean | ~315,059,300 MWh |
| Std Dev | ~31,899,940 MWh |
| Min | ~262,400,600 MWh |
| 25th Percentile | ~291,437,900 MWh |
| Median (50%) | ~308,153,300 MWh |
| 75th Percentile | ~332,148,500 MWh |
| Max | ~392,436,034 MWh |

---

## 🎯 Objectives

1. **Analyze** historical electricity demand data to identify underlying **trends**, **seasonality**, and **cyclic patterns**.
2. **Test for stationarity** using the Augmented Dickey-Fuller (ADF) test and transform the series where required.
3. **Build and compare** four forecasting models spanning classical statistics and deep learning:
   - ARIMA (AutoRegressive Integrated Moving Average)
   - SARIMA (Seasonal ARIMA)
   - Facebook Prophet
   - LSTM (Long Short-Term Memory Neural Network)
4. **Evaluate** each model's performance using consistent error metrics (MAPE, RMSE, MAE).
5. **Forecast** future electricity demand for the **next 36 months** (Jan 2024 – Dec 2026).

---

## 🧩 Methodology & Notebook Workflow

The Jupyter notebook contains **50 cells** (43 code + 7 markdown) organized into the following analytical phases:

### Phase 1 — Data Loading & Preprocessing (Cells 1–5)

- **Library imports**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `pmdarima`, `sklearn`, `tensorflow/keras`, `prophet`
- **Data loading**: Read `Elec_sales.xlsx` using `openpyxl` engine
- **Data cleaning**: Removed commas from numerical fields and cast 14 columns to `float`
- **Datetime indexing**: Combined `Year` and `Month` into a proper `datetime` index (`pd.to_datetime`)
- **Missing value audit**: Confirmed **zero null values** across all 22 columns

### Phase 2 — Exploratory Data Analysis (Cells 6–8)

- **Multi-variable time series plots**: Visualized all 20 numeric features over time (Revenue, Sales, Customers, Price for each sector + totals)
- **Descriptive statistics**: Computed `describe()` for `Sales_total`
- **Target series plot**: Dedicated line chart for `Sales_total` over the full 14-year period, revealing a clear **upward trend** with **strong annual seasonality** (peaks in summer months driven by cooling demand)

### Phase 3 — Stationarity Testing & Differencing (Cells 9–14)

- **ADF Test on raw series**:
  - ADF Statistic: **−2.341** | p-value: **0.159**
  - Conclusion: **Non-stationary** (p > 0.05)
- **First-order differencing** (`diff()`): Applied to remove the trend component
- **ADF Test on differenced series**:
  - ADF Statistic: **−6.974** | p-value: **8.53e-10**
  - Conclusion: **Stationary** (p ≪ 0.05) ✅
- **ACF & PACF analysis**: Plotted for the differenced series to identify candidate `p` and `q` orders for ARIMA

### Phase 4 — ARIMA Modeling (Cells 15–30)

- **Grid search** over `p ∈ {0,1,2}`, `d ∈ {0,1}`, `q ∈ {0,1,2}` on the differenced series — best config: **ARIMA(2, 0, 2)** with AIC = 6067.73
- **auto_arima** (via `pmdarima`) confirmed optimal non-seasonal parameters
- **Model fitting**: ARIMA(2, 0, 2) on differenced data with full model summary
- **Residual diagnostics**:
  - Residual time plot (centered around zero)
  - Histogram with KDE (approximately normal)
  - QQ plot (normality check)
  - ACF & PACF of residuals (no significant autocorrelation remaining)
- **36-month forecast**: Differenced forecasts re-integrated to original scale via `cumsum()`
- **Train/test evaluation** (80/20 split):

| Metric | Value |
|---|---|
| RMSE | 26,046,784 |
| MAE | 20,938,192 |
| **MAPE** | **6.36%** |

### Phase 5 — SARIMA Modeling (Cells 31–40)

- **Seasonal decomposition** (`seasonal_decompose`, period=12): Extracted trend, seasonal, and residual components — confirmed strong **12-month periodicity**
- **auto_arima with `seasonal=True`**: Explored numerous (P,D,Q,s) combinations — best: **ARIMA(1, 0, 2)(2, 1, 2)[12]** with AIC = 5423.11
- **Manual SARIMA fit**: `SARIMAX(order=(1,0,2), seasonal_order=(2,1,2,12))`
- **Residual analysis**: ACF & PACF of SARIMA residuals showed negligible autocorrelation
- **36-month forecast with 95% confidence intervals**
- **Train/test evaluation** (last 12 months held out):

| Metric | Value |
|---|---|
| RMSE | 5,315,865 |
| MAE | 3,397,019 |
| **MAPE** | **1.11%** |

### Phase 6 — Facebook Prophet (Cells 41–45)

- **Data reformatting**: Prepared `ds` (datetime) and `y` (Sales_total) columns per Prophet's API
- **Model fitting**: Default Prophet with additive components
- **Forecast visualization**: Historical + future predictions with uncertainty bands
- **Train/test evaluation** (last 12 months held out):

| Metric | Value |
|---|---|
| RMSE | 46,262,329 |
| MAE | 36,851,586 |
| **MAPE** | **11.69%** |

### Phase 7 — LSTM Neural Network (Cells 46–48)

- **Data scaling**: MinMaxScaler (0–1 normalization) on `Sales_total`
- **Sequence creation**: Sliding window of **12 months** to predict the next month
- **Model architecture**:
  ```
  Sequential([
      LSTM(50 units, activation='relu', input_shape=(12, 1)),
      Dense(1)
  ])
  ```
- **Training**: 100 epochs, batch size 16, optimizer = Adam, loss = MSE
- **Train/test evaluation** (last 12 months held out):

| Metric | Value |
|---|---|
| RMSE | 8,802,357 |
| MAE | 7,254,553 |
| **MAPE** | **2.46%** |

- **36-month future forecast** (Jan 2024 – Dec 2026): Generated via iterative autoregressive prediction, with the model successfully reproducing the seasonal demand pattern

---

## 📈 Results — Model Comparison

| Model | MAPE (%) | RMSE | MAE | Test Horizon | Key Insights |
|---|---|---|---|---|---|
| **ARIMA(2,0,2)** | 6.36 | 26,046,784 | 20,938,192 | 20% hold-out | Captured trend on differenced data but **missed seasonal patterns** |
| **SARIMA(1,0,2)(2,1,2)[12]** | **1.11** | **5,315,865** | **3,397,019** | Last 12 months | **Best MAPE** — effectively modeled both trend and 12-month seasonality |
| **Prophet** | 11.69 | 46,262,329 | 36,851,586 | Last 12 months | Smooth additive decomposition but **struggled with this data's amplitude** |
| **LSTM (50 units)** | 2.46 | 8,802,357 | 7,254,553 | Last 12 months | Strong non-linear learning; captured seasonal cycles well |

### Key Takeaways

- 📊 **SARIMA achieved the lowest MAPE (1.11%)**, demonstrating that explicitly modeling 12-month seasonal orders is highly effective for this electricity demand series.
- 🧠 **LSTM (MAPE 2.46%)** performed as the **best non-parametric model**, successfully learning temporal dependencies without explicit seasonal specification.
- ⚠️ **ARIMA without seasonal terms** produced reasonable results (6.36%) but underperformed compared to seasonal-aware methods, confirming that seasonality is a dominant driver of electricity demand.
- 📉 **Prophet underperformed** on this dataset (11.69% MAPE), likely due to the strong periodic amplitude that its default additive model could not fully capture.
- ✅ The **LSTM model was used for the final 36-month forecast** (Jan 2024 – Dec 2026), producing realistic seasonal demand patterns.

---

## 🧠 Tools & Technologies

| Category | Technologies |
|---|---|
| **Language** | Python 3.x |
| **Data Handling** | Pandas, NumPy, openpyxl |
| **Visualization** | Matplotlib, Seaborn |
| **Statistical Modeling** | Statsmodels (ARIMA, SARIMAX, seasonal_decompose, ADF test, ACF/PACF) |
| **Automated Model Selection** | pmdarima (`auto_arima`) |
| **Additive Forecasting** | Facebook Prophet |
| **Deep Learning** | TensorFlow / Keras (LSTM, Dense layers) |
| **Evaluation** | Scikit-learn (MAE, MSE), custom MAPE |
| **Diagnostics** | QQ plots (`statsmodels.graphics.gofplots`) |

---

## 🚀 How to Run

### Prerequisites

Ensure Python 3.8+ is installed, then install dependencies:

```bash
pip install pandas numpy matplotlib seaborn openpyxl statsmodels pmdarima scikit-learn prophet tensorflow
```

### Run the Notebook

```bash
jupyter notebook "Electricity Demand Forecasting.ipynb"
```

### View Outputs

The notebook produces the following outputs:
- 📈 Time series line charts for all 20+ variables
- 📊 ACF / PACF correlograms for parameter selection
- 🔍 Residual diagnostic plots (time series, histogram, QQ plot)
- 🔮 36-month demand forecasts with confidence intervals (SARIMA) and point forecasts (ARIMA, LSTM)
- 📋 Model comparison metrics (MAPE, RMSE, MAE)

---

## 🔮 Future Work

- 📌 Incorporate **exogenous variables** (temperature, humidity, holidays, economic indicators) into SARIMAX/LSTM for improved accuracy.
- 🧪 Experiment with **Bi-directional LSTM**, **GRU**, and **Temporal Convolutional Networks (TCN)** architectures.
- 🔧 Apply **hyperparameter tuning** (learning rate, number of layers/units, dropout) using grid/random/Bayesian search.
- 🌐 Deploy a real-time **forecasting dashboard** or **REST API** for operational electricity demand prediction.
- 📊 Add **cross-validation** with expanding/sliding window strategies for more robust model evaluation.
- ⚙️ Integrate **Prophet's advanced features** (custom seasonalities, holiday effects, regressors) to improve its performance on this dataset.

---

## 📝 License

This project is intended for educational and portfolio demonstration purposes.
