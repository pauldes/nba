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

def scale_per_value_of(data, selected_cat_features, selected_num_features, scale_per_value_of, min_max_scaler=True):
    if selected_num_features is None or len(selected_num_features)==0:
        raise NotImplementedError("Need at least 1 numerical feature")
    X_num = standardize(data[selected_num_features], fit_on=None, fit_per_values_of=scale_per_value_of, min_max_scaler=min_max_scaler)
    if selected_cat_features is not None and len(selected_cat_features)>0:
        X_cat = pandas.get_dummies(data[selected_cat_features]) #, drop_first=True)
        X_processed = pandas.concat([X_num, X_cat], axis=1)
        X_raw = data[selected_num_features + selected_cat_features]
    else:
        X_processed = X_num
        X_raw = data[selected_num_features]
    return X_processed, X_raw