# Electricity-Demand-Forecasting

📘 **Overview**:

Forecasted monthly electricity demand to support operational planning and strategic decision-making.
Performed complete time series analysis, model comparison, and multi-year forecasting using both classical and deep learning approaches.

🎯 **Objectives**:

Analyze historical electricity demand data to identify patterns and seasonality.

Test for stationarity and transform the data where necessary.

Build and compare multiple forecasting models (ARIMA, SARIMA, Prophet, LSTM).

Forecast future demand for the next 36 months.

🧩 **Approach**

Data Preprocessing: Cleaned and structured time-indexed monthly data.

Decomposition: Extracted Trend, Seasonal, and Residual components.

Stationarity Check: Applied ADF Test and First Differencing to stabilize the series.

Parameter Tuning: Selected p, d, q parameters using ACF and PACF plots.

**Model Building**:

ARIMA / SARIMA: For linear trend-seasonal modeling.

Prophet: For flexible, additive trend-seasonality modeling.

LSTM: For learning long-term non-linear temporal dependencies.

Evaluation: Compared model performances using MAPE metric.

Forecasting: Extended LSTM model predictions for next 36 months.

📊 **Results**
Model	MAPE (%)	Key Insights
ARIMA	4.82	Captured trend but weak seasonal fit
SARIMA	3.95	Improved seasonal pattern modeling
Prophet	3.21	Smooth forecasts with interpretability
LSTM	2.46	Best performance, strong long-term learning

📈 The LSTM model achieved the lowest MAPE of 2.46%, indicating superior accuracy in long-term forecasting.

🧠 **Tools & Technologies**

Python, NumPy, Pandas, Matplotlib, Seaborn

Statsmodels (ARIMA, SARIMA)

Facebook Prophet

TensorFlow / Keras (LSTM)

Scikit-learn



**Install Dependencies**:

pip install -r requirements.txt


Run Jupyter Notebook:

jupyter notebook notebooks/Electricity_Demand_Forecasting.ipynb


View Outputs:

Forecast visualizations

Model comparison metrics

36-month demand forecast

🚀 Future Work Possible:

Include exogenous factors like temperature, holidays, and industrial output.

Experiment with Bi-LSTM and Temporal Convolutional Networks (TCN).

Develop a dashboard or API for real-time electricity demand forecasting.
