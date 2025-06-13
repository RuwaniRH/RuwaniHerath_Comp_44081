import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def evaluate_forecast(actual, predicted, lower=None, upper=None, alpha=0.05):
    """
    Parameters:
    - actual: pd.Series of actual observed values
    - predicted: pd.Series of predicted values
    - lower: pd.Series of lower bound of prediction interval
    - upper: pd.Series of upper bound of prediction interval
    - alpha: significance level (e.g., 0.05 for 95% interval)

    Returns:
    - dict with RMSE, MAE, and coverage 
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    metrics = {
        'RMSE': rmse,
        'MAE': mae
    }

    if lower is not None and upper is not None:
        coverage = ((actual >= lower) & (actual <= upper)).mean()
        metrics[f'Coverage @ {(1 - alpha) * 100:.0f}%'] = coverage

    return metrics


def plot_forecast(df, forecast_df, target_col='y', model_name='Model'):
    """
    Parameters:
    - df: DataFrame with actuals (must contain 'ds' and target_col)
    - forecast_df: Prophet-style forecast DataFrame (must contain 'ds', 'yhat', 'yhat_lower', 'yhat_upper')
    - target_col: column name of actual values in df
    - model_name: title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df[target_col], label='Actual', color='black')
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', color='blue')
    plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                     color='blue', alpha=0.3, label='Confidence Interval')
    plt.title(f'{model_name} Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    
    plt.savefig(f'output/{model_name.lower().replace(" ", "_")}_forecast.png')  # e.g., output/redemption_model_forecast.png
    plt.close()


def rolling_origin_backtest(df, model_class, target, forecast_horizon=30, min_train_days=365):
    """
    Parameters:
    - df: full DataFrame with datetime index
    - model_class: class implementing Prophet-style interface (fit, forecast)
    - target: target column name
    - forecast_horizon: days to forecast each time
    - min_train_days: initial training size

    Returns:
    - DataFrame with actuals, predictions, intervals, and evaluation metrics
    """
    results = []

    for start in range(0, len(df) - min_train_days - forecast_horizon, forecast_horizon):
        train = df.iloc[start:start + min_train_days]
        test = df.iloc[start + min_train_days:start + min_train_days + forecast_horizon]

        model = model_class(train, target)
        model.fit()
        forecast = model.forecast(forecast_horizon)

        merged = test.reset_index().merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
        merged['actual'] = test[target].values

        eval_metrics = evaluate_forecast(
            actual=merged['actual'],
            predicted=merged['yhat'],
            lower=merged['yhat_lower'],
            upper=merged['yhat_upper']
        )

        eval_metrics['start_date'] = test.index[0]
        results.append(eval_metrics)

    return pd.DataFrame(results)
