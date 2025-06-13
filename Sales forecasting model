from prophet import Prophet
import pandas as pd

class SalesForecastModel:
    def __init__(self, df, target='Sales Count'):
        self.df = df
        self.target = target
        self.model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)

    def prepare_data(self):
        df_prophet = self.df[[self.target]].reset_index()
        df_prophet.columns = ['ds', 'y']
        return df_prophet

    def fit(self):
        df_train = self.prepare_data()
        self.model.fit(df_train)

    def forecast(self, days=30):
        future = self.model.make_future_dataframe(periods=days)
        forecast = self.model.predict(future)
        return forecast

    def plot_forecast(self, forecast):
        self.model.plot(forecast)

    def run(self):
        self.fit()
        forecast = self.forecast()
        self.plot_forecast(forecast)
        return forecast

