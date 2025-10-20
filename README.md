# stock-price-forecasting
Machine learning project for forecasting Apple stock prices. Demonstrates data collection, feature engineering, Random Forest modeling, and visualization of financial time series.

# Stock Price Forecasting with Random Forest (AAPL)

## Overview
This project demonstrates a preliminary attempt to forecast **next-day stock prices** for **Apple Inc. (AAPL)** using **Random Forest Regression**.  
It involves downloading historical stock data, engineering technical features, training a machine learning model, and visualizing the results.

This project is part of my portfolio for **summer internship applications in quantitative finance and machine learning**.

---

## Features Created
- `Close` – daily closing price  
- `Return` – daily percentage change in price  
- `MA5`, `MA10`, `MA20`, `MA50`, `MA100` – moving averages for short and long-term trends  
- `Volatility` – rolling standard deviation of returns (window=10)  
- `Momentum` – price momentum over 5 days  

These features capture **trend, volatility, and momentum** patterns in stock prices.

---

## Model
- **Algorithm:** Random Forest Regressor  
- **Parameters:** 100 estimators, random state = 42  
- **Target variable:** next day `Close` price  

### Evaluation Metrics
- **Mean Squared Error (MSE):** [computed value]  
- **Mean Absolute Error (MAE):** [computed value]  
- **MAE percentage:** [computed value]  

> Note: The model shows preliminary predictions and tends to **underestimate higher prices** due to the limitations of Random Forest in extrapolating trends.

---

## Visualizations
- **Line Plot:** Actual vs Predicted Prices  
- **Histogram:** Distribution of daily returns  
- **Scatter Plot:** Predicted vs Actual Prices  
- **Correlation Heatmap:** Shows relationships between features  
- **Boxplot of Features:** Detects outliers and data distribution  
- **Pairplot:** Pairwise relationships between features  

These plots help to understand the data, feature relationships, and model performance.

---

## Limitations and Next Steps
1. Random Forest may **struggle with trend extrapolation** in stock prices.  
2. Some features (like MA50 and MA100) are **highly correlated**, which may affect model performance.  
3. Future improvements could include:
   - Predicting **returns instead of absolute prices**  
   - Using **Gradient Boosting or Neural Networks**  
   - Creating **lag features** to capture temporal patterns  
   - Hyperparameter tuning for better accuracy  

## Technologies used
- Python 3.11
- pandas, yfinance
- scikit-learn
- matplotlib, seaborn
