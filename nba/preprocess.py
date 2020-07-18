import pandas

def get_numerical_columns(dataframe):
    return dataframe.select_dtypes(include="number")

def get_categorical_columns(dataframe):
    return dataframe.select_dtypes(exclude="number")