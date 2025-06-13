import os
from data_loader import load_data
from features import add_time_features, add_lag_features, add_rolling_features
from Improved_Forecasting_model import ImprovedRedemptionModel
from Sales_forecasting_model import SalesForecastModel
from Evaluate import evaluate_forecast, plot_forecast
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

df = load_data('./Toronto Island Ferry Ticket Counts.csv')

df = add_time_features(df)
df = add_lag_features(df, target_col='Redemption Count')
df = add_lag_features(df, target_col='Sales Count')
df = add_rolling_features(df, target_col='Redemption Count')
df = add_rolling_features(df, target_col='Sales Count')
df.dropna(inplace=True)

# --- Redemption Model ---
print("Running Redemption Forecast Model...")
redemption_model = ImprovedRedemptionModel(df, target='Redemption Count')
redemption_forecast = redemption_model.run()


forecast_redemption = redemption_model.forecast(30)
actual = df[['Redemption Count']].reset_index().rename(columns={'Timestamp': 'ds', 'Redemption Count': 'y'})
merged = actual.merge(forecast_redemption[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
metrics = evaluate_forecast(
    merged['y'],
    merged['yhat'],
    merged['yhat_lower'],
    merged['yhat_upper']
)

plt.figure(figsize=(12, 6))
plt.plot(merged['ds'], merged['y'], label='Actual', color='black')
plt.plot(merged['ds'], merged['yhat'], label='Forecast', color='blue')
plt.fill_between(merged['ds'], merged['yhat_lower'], merged['yhat_upper'], color='blue', alpha=0.3)
plt.title('Redemption Model Forecast')
plt.xlabel('Date')
plt.ylabel('Redemption Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/redemption_model_forecast.png')
plt.close()

print("Redemption model metrics:", metrics)

# --- Sales Model ---
print("Running Sales Forecast Model...")
sales_model = SalesForecastModel(df, target='Sales Count')
sales_forecast = sales_model.run()


forecast_sales = sales_model.forecast(30)
actual = df[['Sales Count']].reset_index().rename(columns={'Timestamp': 'ds', 'Sales Count': 'y'})
merged = actual.merge(forecast_sales[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
metrics = evaluate_forecast(
    merged['y'],
    merged['yhat'],
    merged['yhat_lower'],
    merged['yhat_upper']
)

plt.figure(figsize=(12, 6))
plt.plot(merged['ds'], merged['y'], label='Actual', color='black')
plt.plot(merged['ds'], merged['yhat'], label='Forecast', color='green')
plt.fill_between(merged['ds'], merged['yhat_lower'], merged['yhat_upper'], color='green', alpha=0.3)
plt.title('Sales Model Forecast')
plt.xlabel('Date')
plt.ylabel('Sales Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/sales_model_forecast.png')
plt.close()

print("Sales model metrics:", metrics)
