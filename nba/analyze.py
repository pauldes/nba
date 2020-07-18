import numpy
import seaborn

def get_column_pairs_correlation(dataframe):
    corr = dataframe.corr().abs()
    s = corr.unstack()
    so = s.sort_values(ascending=False)
    so = so[so<1.000]
    return so

def get_columns_correlation_with_target(dataframe, target_column):
    corr = dataframe.corr()
    res = corr[target_column].abs().sort_values(ascending=False)
    res = res[res<1.000]
    return res

def plot_correlation_heatmap(dataframe, corner=True):
    # Compute the correlation matrix
    corr = dataframe.corr()
    # Generate a mask for the upper triangle
    if corner:
        mask = numpy.triu(numpy.ones_like(corr, dtype=numpy.bool))
    else:
        mask=None
    # Set up the matplotlib figure
    #f, ax = pyplot.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = seaborn.diverging_palette(240, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    seaborn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")