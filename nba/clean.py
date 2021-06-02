import pandas
from scipy import stats
import numpy


def safe_drop(dataframe, cols):
    cols = [col for col in cols if col in dataframe.columns]
    dataframe = dataframe.drop(cols, axis="columns", inplace=False)
    return dataframe


def drop_columns_by_nan_ratio(dataframe, ratio: float):
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Ratio must be between 0 and 1")
    else:
        ratios = dataframe.isna().sum(axis="index") / len(dataframe)
        ratios = ratios[ratios > ratio]
        cols = ratios.index.tolist()
        print("Dropping", cols, "...")
        dataframe = dataframe.drop(cols, axis="columns", inplace=False)
        return dataframe


def pretty_print_df_values(dataframe_values):
    entries = []
    for v in dataframe_values:
        entry = " :: ".join([str(vi) for vi in v])
        entries.append(entry)
    print("\n".join(entries))


def clean():

    data = pandas.read_csv("../data/training/all_consolidated_raw.csv")
    print(data.isna().sum(axis="index").sort_values(ascending=False).head(10))
    data = drop_columns_by_nan_ratio(data, 0.1)

    data["MVP_PODIUM"] = data["MVP_PODIUM"].fillna(False)
    data["MVP_CANDIDATE"] = data["MVP_CANDIDATE"].fillna(False)
    data["MVP_WINNER"] = data["MVP_WINNER"].fillna(False)
    data["MVP_VOTES_SHARE"] = data["MVP_VOTES_SHARE"].fillna(0.0)

    print(data.isna().sum(axis="index").sort_values(ascending=False).head(10))

    cols_with_nan_all = data.isna().sum(axis="index").sort_values(ascending=False)
    cols_with_nan_all = cols_with_nan_all[cols_with_nan_all > 0]
    cols_with_nan_all = cols_with_nan_all.index.tolist()
    mvp_candidates = data[data["MVP_CANDIDATE"] == True]
    cols_with_nan_candidates = (
        mvp_candidates.isna().sum(axis="index").sort_values(ascending=False)
    )
    cols_with_nan_candidates = cols_with_nan_candidates[cols_with_nan_candidates > 0]
    cols_with_nan_candidates = cols_with_nan_candidates.index.tolist()
    for col in cols_with_nan_all:
        if col not in cols_with_nan_candidates:
            print("Dropping NAN on col", col, "because no MVP candidates have NAN...")
            data = data.dropna(subset=[col], axis="index")
        else:
            print(
                "Not dropping NAN on col",
                col,
                "because some MVPs candidates have NAN too",
            )

    print(data.isna().sum(axis="index").sort_values(ascending=False).head(10))
    print(data[data.isna().any(axis="columns")].SEASON.unique())

    cols_to_fill = data.isna().sum(axis="index")
    cols_to_fill = cols_to_fill[cols_to_fill > 0]
    cols_to_fill = cols_to_fill.index.tolist()

    # Median of the same position depending on MVP candidate caliber

    for col in cols_to_fill:
        print("Filling", col, "...")
        for is_mvp_candidate in True, False:
            for position in data.POS.unique():
                reference = data[
                    (data["MVP_CANDIDATE"] == is_mvp_candidate)
                    & (data["POS"] == position)
                ][col]
                reference_median = reference[reference.notna()].median()
                print(
                    "Stat :",
                    col,
                    "| Position :",
                    position,
                    "| MVP candidate :",
                    is_mvp_candidate,
                    " | Median :",
                    reference_median,
                )
                data.loc[
                    (data["MVP_CANDIDATE"] == is_mvp_candidate)
                    & (data["POS"] == position)
                    & (data[col].isna()),
                    col,
                ] = reference_median

    print(data.isna().sum(axis="index").sort_values(ascending=False).head(10))

    for season in data.SEASON.unique():
        for col in data.columns:
            if col not in [
                "MVP_VOTES_SHARE",
                "MVP_WINNER",
                "MVP_PODIUM",
                "MVP_CANDIDATE",
            ]:
                try:
                    outliers = data[
                        (stats.zscore(data[col]) > 7) & (data.SEASON == season)
                    ][["player_season_team", col, "MVP_CANDIDATE"]]
                    outliers_mvp_candidates = outliers[
                        outliers["MVP_CANDIDATE"] == True
                    ][["player_season_team", col]]
                    outliers_non_mvp_candidates = outliers[
                        outliers["MVP_CANDIDATE"] == False
                    ][["player_season_team", col]]
                    if len(outliers) > 0:
                        print()
                        print("Outliers for", col, "in season", season, ":")
                        pretty_print_df_values(outliers.values)
                    if len(outliers_non_mvp_candidates) > 0:
                        invalid_id = outliers_non_mvp_candidates.index.tolist()
                        data = data.loc[~data.index.isin(invalid_id)]
                except TypeError:
                    pass

    # Two methods :
    # data = data.T.drop_duplicates(how="all", keep="first").T
    # data = data.drop(['MP_per_100poss'], axis='columns')

    data.to_csv("../data/training/all_consolidated_final.csv", index=False)
