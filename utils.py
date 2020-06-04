import numpy as np


def create_time_features(df, date_var=None, features=None, label=None):
    """
    Creates time series features from datetime variable
    """
    if features is None:
        features = []
    df['date'] = df[date_var]
    df['hour'] = df['date'].dt.hour
    # df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    # df['dayofyear'] = df['date'].dt.dayofyear
    # df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    # X = df[['hour','dayofweek','quarter','month','year','dayofyear','dayofmonth','weekofyear']+features]
    X = df[['hour', 'quarter', 'month', 'year', 'weekofyear'] + features]

    if label:
        y = df[label]
        return X, y
    return X


def mape(a, b):
    mask = a != 0
    return np.mean(np.abs((a - b) / a)[mask]) * 100


def smape(a, b):
    return np.mean(np.abs(a - b) / (np.abs(a) + np.abs(b))) * 100


def vwape(a, b):
    return (np.sum(np.abs(a - b)) / np.sum(a)) * 100


def mean_absolute_error(a, b):
    mask = a != 0
    return np.mean(np.abs((a - b))[mask])


def metrics(a, b):
    print(f'MAPE: {round(mape(a, b), 1)} %')
    print(f'SMAPE: {round(smape(a, b), 1)} %')
    print(f'VWAPE: {round(vwape(a, b), 1)} %')
    print(f'MAE: {round(mean_absolute_error(a, b), 1)}')
