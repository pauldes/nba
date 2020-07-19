import numpy
import seaborn

def get_columns_with_inter_correlations_under(dataframe, treshold):
    keep = dataframe.columns
    data_curr = dataframe.copy()
    initial_num_cols = len(data_curr.columns)
    corr = get_column_pairs_correlation(data_curr)
    corr = corr[corr>treshold]
    while len(corr) > 0:
        col_to_drop = corr.index.values[0][1]
        print("Dropping", col_to_drop, "which is correlated with", corr.index.values[0][0])
        data_curr = data_curr.drop(col_to_drop, axis='columns')
        corr = get_column_pairs_correlation(data_curr)
        corr = corr[corr>treshold]
    res = data_curr.columns
    print("Reduced number of columns from", initial_num_cols, "to", len(res))
    return res

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

def pairplot_columns(dataframe, columns, color_by):
    seaborn.pairplot(dataframe, hue=color_by, x_vars=columns, y_vars=columns, corner=True)

def plot_columns_against_target(dataframe, columns, target_column):
    seaborn.pairplot(dataframe, x_vars=[target_column], y_vars=columns, corner=True)

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