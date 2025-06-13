def add_time_features(df):
    df = df.copy()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_week'] = df.index.dayofweek
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def add_lag_features(df, target_col='Redemption Count', lags=[1, 7]):
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    return df

def add_rolling_features(df, target_col='Redemption Count', windows=[7, 30]):
    for win in windows:
        df[f'{target_col}_rollmean{win}'] = df[target_col].rolling(win).mean()
    return df
