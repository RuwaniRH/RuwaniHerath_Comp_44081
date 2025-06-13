from data_loader import load_data
from features import add_time_features, add_lag_features, add_rolling_features
from Improved_Forecasting_model import ImprovedRedemptionModel
from Sales_forecasting_model import SalesForecastModel
from Evaluate import evaluate_forecast, plot_forecast

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

# --- Sales Model ---
print("Running Sales Forecast Model...")
sales_model = SalesForecastModel(df, target='Sales Count')
sales_forecast = sales_model.run()


forecast_redemption = redemption_model.forecast(30)
actual = df[['Redemption Count']].reset_index().rename(columns={'Timestamp': 'ds', 'Redemption Count': 'y'})
merged = actual.merge(forecast_redemption[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
metrics = evaluate_forecast(
    merged['y'],
    merged['yhat'],
    merged['yhat_lower'],
    merged['yhat_upper']
)
plot_forecast(merged, merged, target_col='y', model_name='Redemption Model')
print(metrics)


forecast_sales = sales_model.forecast(30)
actual = df[['Sales Count']].reset_index().rename(columns={'Timestamp': 'ds', 'Sales Count': 'y'})
merged = actual.merge(forecast_sales[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
metrics = evaluate_forecast(
  merged['y'], 
  merged['yhat'], 
  merged['yhat_lower'], 
  merged['yhat_upper']
)
plot_forecast(merged, merged, target_col='y', model_name='Sales Model')
print(metrics)



