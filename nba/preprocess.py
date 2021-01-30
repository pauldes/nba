import random

import pandas
import numpy
import sklearn

def standardize(dataframe, fit_on=None, fit_per_values_of=None, min_max_scaler=False):
    if min_max_scaler:
        scaler = sklearn.preprocessing.MinMaxScaler()
    else:
        scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)
    if fit_on is not None and fit_per_values_of is not None:
        raise NotImplementedError
    if fit_on is None:
        fit_on = dataframe.copy()
    scaled = dataframe.copy()
    if fit_per_values_of is not None:
        series = fit_per_values_of.copy()
        for unique in series.unique():
            curr_index = series[series==unique].index
            df_subset = dataframe.loc[curr_index, :]
            scaler.fit(df_subset[df_subset.columns])
            scaled.loc[curr_index, scaled.columns] = scaler.transform(df_subset[df_subset.columns])
    else:
        scaler.fit(fit_on[fit_on.columns])
        scaled[scaled.columns] = scaler.transform(dataframe[dataframe.columns]) 
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
    share_int = int(share*len(series.unique()))
    seasons = series.sample(share_int).tolist()
    print(share_int)
    return seasons