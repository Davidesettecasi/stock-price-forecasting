import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Download historical stock data for Apple Inc. (AAPL)
ticker = "AAPL"  # Ticker symbol for Apple
data = yf.download(ticker, start="2018-01-01", end="2023-12-31")
data = data[['Close']]  # Keep only the closing price


data['Return'] = data['Close'].pct_change()
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA100'] = data['Close'].rolling(window=100).mean()
data['Volatility'] = data['Return'].rolling(window=10).std()
data['Momentum'] = data['Close'] / data['Close'].shift(5) - 1
data=data.dropna()

# Prepare features and target variable
data['Target'] = data['Close'].shift(-1)  # Predict next day's closing price
data = data.dropna()  # Drop rows with NaN values
X = data[['Close', 'Return', 'MA5', 'MA10', 'MA20', 'MA100', 'Volatility', 'Momentum']]
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mae_percent = mae / y_test.mean() * 100

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print(f"MAE percentuale: {mae_percent:.2f}%")

# Plot actual vs predicted prices 
plt.figure(figsize=(12,6))
sns.lineplot(x=y_test.index, y=y_test, label='Actual')
sns.lineplot(x=y_test.index, y=y_pred, label='Predicted')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#Historigram
plt.figure(figsize=(8,5))
sns.histplot(data['Return'], bins=50, kde=True)
plt.title('Distribution of Daily Returns')
plt.show()

#scatter plot predicted vs actual
plt.figure(figsize=(8,8))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # diagonale perfetta
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()

#Correlation heatmap
plt.figure(figsize=(10,8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#Feature boxplot
plt.figure(figsize=(10,6))
sns.boxplot(data=data[['Return','Volatility','Momentum']])
plt.title('Boxplot of Features')
plt.show()

#Feature pairplot
sns.pairplot(data[['Close', 'Return', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'Volatility', 'Momentum']])
plt.show()
