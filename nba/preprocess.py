import random

import pandas
import numpy
import sklearn

def standardize(dataframe, fit_on=None, fit_per_values_of=None):
    scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)
    if fit_on is None:
        fit_on = dataframe
    scaled = dataframe.copy()
    scaler.fit(fit_on[fit_on.columns])
    scaled[scaled.columns] = scaler.fit_transform(dataframe[dataframe.columns]) 
    return scaled

def get_numerical_columns(dataframe):
    return dataframe.select_dtypes(include="number").columns

def get_categorical_columns(dataframe):
    return dataframe.select_dtypes(exclude="number").columns

def natural_log_transform(series):
    return numpy.log(series+1)

def exp_transform(series):
    return numpy.exp(series)

def select_random_unique_values(series, share:float):
    seasons = series.unique()
    share_int = int(share*len(seasons))
    seasons = random.choices(seasons, k=share_int)
    print(share_int)
    return seasons