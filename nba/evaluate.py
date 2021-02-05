import pandas
import numpy

def softmax(series):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(series - numpy.max(series))
    return e_x / e_x.sum()

def share(series):
    return series / series.sum()