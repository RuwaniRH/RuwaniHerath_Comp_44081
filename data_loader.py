import pandas as pd

def load_data(path):
    df = pd.read_csv(path,
                     dtype={'_id': int, 'Redemption Count': int, 'Sales Count': int},
                     parse_dates=['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    df.set_index('Timestamp', inplace=True)
    return df.resample('D').sum()
