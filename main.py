from data_loader import load_data
from features import add_time_features, add_lag_features, add_rolling_features
from model_redemption import ImprovedRedemptionModel
from model_sales import SalesForecastModel

df = load_data('./data/Toronto Island Ferry Ticket Counts.csv')

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
