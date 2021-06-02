import datetime
import os
import errno

import streamlit as st
import pandas
import joblib
import shap
import sklearn
from matplotlib import pyplot

from nba import br_extractor, preprocess, evaluate

# Constants
PAGE_PREDICTIONS = "Current year predictions"
PAGE_PERFORMANCE = "Model performance analysis"
CONFIDENCE_MODE_SOFTMAX = "Softmax-based"
CONFIDENCE_MODE_SHARE = "Percentage-based"

year = datetime.datetime.now().year  # TODO : season and year may be different
month = datetime.datetime.now().month
day = datetime.datetime.now().day

# Page properties
st.set_page_config(
    page_title="Predicting the MVP",
    page_icon=":basketball:",
    layout="centered",
    initial_sidebar_state="auto",
)
pyplot.style.use('dark_background')

# Functions
# @st.cache
def load_model():
    """Load the model.

    Returns:
        sklearn.base.BaseEstimator: The model to use for prediction
    """
    return joblib.load("static/model/model.joblib")


@st.cache(allow_output_mutation=True)
def build_history(day, month, season):
    # TODO use a database or other persistent storage
    # For now files are deleted at each deploy
    folder = "./data/predictions/"
    stats = pandas.DataFrame(columns=["player", "date", "prediction"])
    for filename in os.listdir(folder):
        if str(filename).endswith(".csv"):
            predictions = pandas.read_csv(
                folder + filename,
                header=None,
                index_col=None,
                names=["player", "prediction"],
                dtype={"player": str, "prediction": float},
            )
            predictions["date"] = pandas.to_datetime(
                filename, format="%Y_%m_%d.csv", errors="ignore"
            )
            stats = stats.append(predictions, sort=False)
    return stats


def create_data_folder(day, month, season):
    day = str(day).rjust(2, "0")
    month = str(month).rjust(2, "0")
    folder = f"./data/current/{season}_{month}_{day}/"
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


@st.cache
def load_player_stats(day, month, season):
    extractor = br_extractor.BRExtractor()
    stats = extractor.get_player_stats(
        subset_by_seasons=[season],
        subset_by_stat_types=["per_game", "per_36min", "per_100poss", "advanced"],
    )
    stats["SEASON"] = stats["SEASON"].astype(int)
    day = str(day).rjust(2, "0")
    month = str(month).rjust(2, "0")
    folder = f"./data/current/{season}_{month}_{day}/"
    filename = "player_stats.csv"
    if filename not in os.listdir(folder):
        stats.to_csv(folder + filename)
    return stats

def is_season_ended():
    """ Find out if current season is over.

    Returns:
        boolean: True if season is over, False if season is in progress.
    """
    return True

@st.cache
def load_team_stats(day, month, season):
    extractor = br_extractor.BRExtractor()
    stats = extractor.get_team_standings(subset_by_seasons=[season])
    stats["SEASON"] = stats["SEASON"].astype(int)
    day = str(day).rjust(2, "0")
    month = str(month).rjust(2, "0")
    folder = f"./data/current/{season}_{month}_{day}/"
    filename = "team_stats.csv"
    if filename not in os.listdir(folder):
        stats.to_csv(folder + filename)
    return stats


@st.cache
def consolidate_stats(team_stats, player_stats, day, month, season):
    stats = player_stats.merge(team_stats, how="inner", on=["TEAM", "SEASON"])
    stats = stats.set_index("PLAYER", drop=True)
    day = str(day).rjust(2, "0")
    month = str(month).rjust(2, "0")
    folder = f"./data/current/{season}_{month}_{day}/"
    filename = "consolidated_stats.csv"
    if filename not in os.listdir(folder):
        stats.to_csv(folder + filename)
    return stats


def load_2020_preds():
    """Load the predictions made by the trained model for the 2020 season.

    Returns:
        pandas.DataFrame: Detailed redictions made for 2020 season
    """
    preds = pandas.read_csv("./static/data/2020_dataset_predictions.csv")
    return preds


def load_test_preds():
    """Load the predictions made by the trained model on the test dataset.

    Returns:
        pandas.DataFrame: Predicted MVP for the test seasons
    """
    preds = pandas.read_csv("./static/data/test_dataset_predictions.csv")
    return preds


def mvp_found_pct(test_dataset_predictions):
    metrics = (
        test_dataset_predictions["Pred. MVP"] == test_dataset_predictions["True MVP"]
    ).sum() / len(test_dataset_predictions)
    metrics = int(metrics * 100)
    return str(metrics) + " %"


def avg_real_mvp_rank(test_dataset_predictions):
    metrics = (test_dataset_predictions["REAL_RANK"]).mean()
    return "%.2f" % metrics


def clean_data(data):
    # TODO : automatically reuse cleaning process from training
    data = data.fillna(0.0)
    data["G"] = data["G"].astype(int)
    data = data[data["G"] >= int(0.4 * data["G"].max())]
    data["CONF_RANK"] = data["CONF_RANK"].astype(int)
    data = data[data["CONF_RANK"] <= 8]
    data["MP"] = data["MP"].astype(float)
    data = data[data["MP"] >= 20.0]
    return data


def save_predictions(series, day, month, year, filter_above=0.01):
    day = str(day).rjust(2, "0")
    month = str(month).rjust(2, "0")
    filename = f"{year}_{month}_{day}.csv"
    folder = "./data/predictions/"
    series = series[series > filter_above]
    if filename not in os.listdir(folder):
        series.to_csv(folder + filename, header=False)


def predict(data: pandas.DataFrame, model: sklearn.base.BaseEstimator):
    """Run the prediction pipeline to compute MVP shares

    Args:
        data (pandas.DataFrame): Input data (current season stats)
        model (sklearn.base.BaseEstimator): Trained model

    Returns:
        [type]: [description]
    """
    # TODO get automatically from training step.. or keep all
    cat = ["POS", "CONF"]
    # TODO get automatically from training step.. or keep all
    num = [
        "2P%",
        "2PA_per_36min",
        "2PA_per_game",
        "2P_per_36min",
        "3P%",
        "3PAR_advanced",
        "3P_per_36min",
        "3P_per_game",
        "AGE",
        "AST%_advanced",
        "AST_per_game",
        "BLK%_advanced",
        "BLK_per_36min",
        "BLK_per_game",
        "BPM_advanced",
        "DBPM_advanced",
        "DRB%_advanced",
        "DRB_per_game",
        "DRTG_per_100poss",
        "DWS_advanced",
        "EFG%_per_game",
        "FG%",
        "FGA_per_100poss",
        "FGA_per_36min",
        "FGA_per_game",
        "FG_per_100poss",
        "FG_per_36min",
        "FG_per_game",
        "FT%",
        "FTA_per_100poss",
        "FTA_per_game",
        "FTR_advanced",
        "FT_per_36min",
        "FT_per_game",
        "G",
        "MP",
        "OBPM_advanced",
        "ORB_per_36min",
        "ORB_per_game",
        "ORTG_per_100poss",
        "OWS_advanced",
        "PER_advanced",
        "PF_per_100poss",
        "PF_per_36min",
        "PF_per_game",
        "PTS_per_100poss",
        "PTS_per_36min",
        "PTS_per_game",
        "STL_per_36min",
        "STL_per_game",
        "TOV%_advanced",
        "TOV_per_36min",
        "TOV_per_game",
        "TRB%_advanced",
        "TRB_per_game",
        "TS%_advanced",
        "USG%_advanced",
        "VORP_advanced",
        "WS/48_advanced",
        "WS_advanced",
        "W",
        "L",
        "W/L%",
        "GB",
        "PW",
        "PL",
        "PS/G",
        "PA/G",
        "CONF_RANK",
    ]
    min_max_scaling = False
    data_processed_features_only, _ = preprocess.scale_per_value_of(
        data, cat, num, data["SEASON"], min_max_scaler=min_max_scaling
    )
    # TODO get automatically from training step
    features = [
        "OBPM_advanced",
        "DRB_per_game",
        "FG_per_100poss",
        "FGA_per_game",
        "TOV_per_game",
        "FG%",
        "PF_per_36min",
        "PS/G",
        "FGA_per_100poss",
        "DBPM_advanced",
        "STL_per_36min",
        "PF_per_100poss",
        "PL",
        "2PA_per_36min",
        "PTS_per_100poss",
        "OWS_advanced",
        "2PA_per_game",
        "BLK_per_game",
        "CONF_RANK",
        "TOV%_advanced",
        "FTA_per_game",
        "TRB%_advanced",
        "W/L%",
        "WS_advanced",
        "DRTG_per_100poss",
        "STL_per_game",
        "EFG%_per_game",
        "TOV_per_36min",
        "PF_per_game",
        "VORP_advanced",
        "FT_per_36min",
        "PER_advanced",
        "USG%_advanced",
        "DRB%_advanced",
        "AST_per_game",
        "W",
        "FTA_per_100poss",
        "FG_per_36min",
        "DWS_advanced",
        "TS%_advanced",
        "FG_per_game",
        "L",
        "WS/48_advanced",
        "BLK_per_36min",
        "G",
        "ORB_per_game",
        "ORB_per_36min",
        "PW",
        "GB",
        "3PAR_advanced",
        "BLK%_advanced",
        "ORTG_per_100poss",
        "PTS_per_36min",
        "FT%",
        "FT_per_game",
        "2P_per_36min",
        "FGA_per_36min",
        "TRB_per_game",
        "2P%",
        "FTR_advanced",
        "PTS_per_game",
        "BPM_advanced",
        "MP",
        "AST%_advanced",
        "POS_C",
        "POS_PF",
        "POS_PG",
        "POS_SF",
        "POS_SG",
        "CONF_EASTERN_CONF",
        "CONF_WESTERN_CONF",
    ]
    X = data_processed_features_only[features]
    preds = model.predict(X)
    return preds, X


@st.cache
def explain(population, sample_to_explain):
    model = load_model()
    explainer = shap.Explainer(model.predict, population, algorithm="auto")
    shap_values = explainer(sample_to_explain)
    return shap_values


def st_shap(plot, height: int = None):
    """Plot a shap-js plot.

    Args:
        plot ([type]): [description]
        height (int, optional): Height of the plot. Defaults to None.
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)


def prepare_history(stats, keep_top_n, confidence_mode, compute_probs_based_on_top_n):
    keep_players = stats.sort_values(by=["date", "prediction"], ascending=False)[
        "player"
    ].to_list()[:keep_top_n]
    for date in stats.date.unique():
        stats.loc[stats.date == date, "rank"] = stats.loc[
            stats.date == date, "prediction"
        ].rank(ascending=False)
        if confidence_mode == CONFIDENCE_MODE_SOFTMAX:
            stats.loc[stats.date == date, "chance"] = (
                evaluate.softmax(stats[stats.date == date]["prediction"]) * 100
            )
            # stats.loc[dataset.rank <= compute_probs_based_on_top_n, "chance"] = evaluate.softmax(dataset[dataset.rank <= compute_probs_based_on_top_n]["prediction"]) * 100
        else:
            stats.loc[stats.date == date, "chance"] = (
                evaluate.share(stats[stats.date == date]["prediction"]) * 100
            )
            # stats.loc[dataset.rank <= compute_probs_based_on_top_n, "chance"] = evaluate.share(dataset[dataset.rank <= compute_probs_based_on_top_n]["prediction"]) * 100
    stats = stats[stats["player"].isin(keep_players)]
    stats = stats.fillna(0.0)
    return stats


def predict_old():
    """A method used adhoc whenever a new model is deployed and we have to run it on the past data to update history."""
    folders = [x[0] for x in os.walk("./data/current/")]
    for folder in folders:
        try:
            folder_date = folder[-10:]
            folder_year = int(folder_date[:4])
            folder_month = int(folder_date[5:7])
            folder_day = int(folder_date[8:])
            current_consolidated_raw = pandas.read_csv(
                folder + "/consolidated_stats.csv", index_col=0
            )
            model = load_model()
            dataset = clean_data(current_consolidated_raw)
            predictions, _ = predict(dataset, model)
            dataset.loc[:, "PRED"] = predictions
            dataset = dataset.sort_values(by="PRED", ascending=False)
            # Replace impossible negative values
            dataset.loc[:, "PRED"] = dataset["PRED"].clip(lower=0.0)
            dataset.loc[:, "PRED_RANK"] = dataset["PRED"].rank(ascending=False)
            save_predictions(dataset["PRED"], folder_day, folder_month, folder_year)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Could not compute predictions for {folder_date} : {e}")


navigation_page = st.sidebar.radio("Navigate to", [PAGE_PREDICTIONS, PAGE_PERFORMANCE])
st.sidebar.markdown(
    """
**How does it work ?**

A statistical regression model is fitted on historical data (player and team stats, and MVP voting results).
It is then used to predict this season MVP based on current data.
The model predicts the *award share* (the number of points a player received for MVP award over the total points of all first-place votes).
This share can then be converted to a chance/probability using various methods (see *Predictions parameters*).
"""
)
st.sidebar.markdown(
    """
*Made by [pauldes](https://github.com/pauldes). Code on [GitHub](https://github.com/pauldes/nba-mvp-prediction).*
"""
)

st.title("🏀 Predicting the MVP")
st.markdown(
    f"""
*Predicting the NBA Most Valuable Player for the {year-1}-{str(year)[-2:]} season using machine learning.*
"""
)

# Init page
create_data_folder(day, month, year)
current_team_stats = load_team_stats(day, month, year)
current_player_stats = load_player_stats(day, month, year)
current_consolidated_raw = consolidate_stats(
    current_team_stats, current_player_stats, day, month, year
).copy()
preds_2020 = load_2020_preds()
preds_test = load_test_preds()
num_test_seasons = len(preds_test)
mvp_found_pct = mvp_found_pct(preds_test)
avg_real_mvp_rank = avg_real_mvp_rank(preds_test)
model = load_model()
dataset = clean_data(current_consolidated_raw)
# Predict
initial_columns = list(dataset.columns)
predictions, model_input = predict(dataset, model)
dataset.loc[:, "PRED"] = predictions
dataset = dataset.sort_values(by="PRED", ascending=False)
# Replace impossible negative values
dataset.loc[:, "PRED"] = dataset["PRED"].clip(lower=0.0)
players_list = dataset.index.to_list()
dataset.loc[:, "PRED_RANK"] = dataset["PRED"].rank(ascending=False)
if is_season_ended:
    pass
else:
    save_predictions(dataset["PRED"], day, month, year)

if navigation_page == PAGE_PREDICTIONS:

    st.header("Current year predictions")
    col1, col2 = st.beta_columns(2)
    col1.subheader("Predicted top 3")
    col2.subheader("Prediction parameters")
    confidence_mode = col2.radio(
        "MVP probability estimation method",
        [CONFIDENCE_MODE_SHARE, CONFIDENCE_MODE_SOFTMAX],
    )
    compute_probs_based_on_top_n = col2.slider(
        "Number of players used to estimate probability",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )
    if confidence_mode == CONFIDENCE_MODE_SOFTMAX:
        dataset.loc[
            dataset.PRED_RANK <= compute_probs_based_on_top_n, "MVP probability"
        ] = (
            evaluate.softmax(
                dataset[dataset.PRED_RANK <= compute_probs_based_on_top_n]["PRED"]
            )
            * 100
        )
    else:
        dataset.loc[
            dataset.PRED_RANK <= compute_probs_based_on_top_n, "MVP probability"
        ] = (
            evaluate.share(
                dataset[dataset.PRED_RANK <= compute_probs_based_on_top_n]["PRED"]
            )
            * 100
        )
    dataset.loc[
        dataset.PRED_RANK > compute_probs_based_on_top_n, "MVP probability"
    ] = 0.0
    dataset["MVP probability"] = dataset["MVP probability"].map("{:,.2f}%".format)
    dataset["MVP rank"] = dataset["PRED_RANK"]
    show_columns = ["MVP probability", "MVP rank"] + initial_columns[:]
    dataset = dataset[show_columns]

    top_3 = dataset["MVP probability"][:3].to_dict()
    emojis = ["🥇", "🥈", "🥉"]

    for n, player_name in enumerate(top_3):
        title_level = "###" + n * "#"
        col1.markdown(
            f"""
        #### {emojis[n]} **{player_name}**

        *{top_3[player_name]} chance to win MVP*
        """
        )

    st.subheader(f"Predicted top {compute_probs_based_on_top_n}")
    st.dataframe(
        data=dataset.head(compute_probs_based_on_top_n), width=None, height=None
    )

    st.subheader("Predictions history")
    col1, col2 = st.beta_columns(2)
    keep_top_n = col2.slider(
        "Number of players to show",
        min_value=3,
        max_value=compute_probs_based_on_top_n,
        value=5,
        step=1,
    )
    variable_to_draw_dict = {
        "MVP chance (%)": "chance",
        "Predicted MVP share": "prediction",
    }
    variable_to_draw = col1.radio(
        "Variable to draw", list(variable_to_draw_dict.keys())
    )
    history = build_history(day, month, year).copy(deep=True)
    prepared_history = prepare_history(
        history, keep_top_n, confidence_mode, compute_probs_based_on_top_n
    )

    st.vega_lite_chart(
        prepared_history,
        {
            "mark": {
                "type": "line",
                "interpolate": "monotone",
                "point": True,
                "tooltip": True,
            },
            "encoding": {
                "x": {"timeUnit": "yearmonthdate", "field": "date"},
                "y": {
                    "field": variable_to_draw_dict[variable_to_draw],
                    "type": "quantitative",
                    "title": variable_to_draw,
                },
                "color": {"field": "player", "type": "nominal"},
            },
        },
        height=400,
        use_container_width=True,
    )

    st.subheader("Predictions explanation")

    model_input_top10 = model_input[model_input.index.isin(players_list[:10])]
    shap_values = explain(model_input_top10, model_input_top10)
    # shap_values = explain(model_input, model_input_top10)
    model_input_top10["player"] = model_input_top10.index
    model_input_top10 = model_input_top10.reset_index(drop=True)

    col1, col2 = st.beta_columns(2)
    selected_player = col1.selectbox("Player", players_list[:10])
    num_features_displayed = col2.slider(
        "Number of features to show", min_value=5, max_value=50, value=10, step=5
    )

    player_index = model_input_top10[model_input_top10.player == selected_player]
    player_index = int(player_index.index[0])
    fig, ax = pyplot.subplots()
    shap.plots.bar(
        shap_values[player_index], max_display=num_features_displayed, show=True
    )
    #shap.plots.waterfall(
    #    shap_values[player_index], max_display=num_features_displayed, show=False
    #)
    pyplot.title(
        f"{num_features_displayed} most impactful features on share prediction for {selected_player}"
    )
    st.markdown(
        """
    Impacts (SHAP values) are relative to the top-10 predicted MVP candidates. These values may not be reliable for categorical variables (as demonstrated [here](https://arxiv.org/pdf/2103.13342.pdf) and [here](https://arxiv.org/pdf/1909.08128.pdf)).
    """
    )
    col1.pyplot(fig, transparent=True, width=None, height=100)

    col21, col22 = col2.beta_columns(2)

    fig, ax = pyplot.subplots()
    shap.summary_plot(shap_values, model_input_top10, plot_type='bar')
    col21.pyplot(fig, transparent=True, width=None, height=100)

    fig, ax = pyplot.subplots()
    shap.summary_plot(shap_values, model_input_top10)
    col21.pyplot(fig, transparent=True, width=None, height=100)


elif navigation_page == PAGE_PERFORMANCE:

    st.header("Model performance analysis")
    st.subheader(f"Test predictions ({num_test_seasons} seasons)")
    st.markdown(
        """
    Predictions of the model on the unseen, test dataset.
    """
    )
    st.markdown(
        f"""
    - **{mvp_found_pct}** of MVPs correctly found
    - Real MVP is ranked in average **{avg_real_mvp_rank}**
    """
    )
    st.dataframe(data=preds_test, width=None, height=None)
    st.subheader("Year 2020")
    st.markdown(
        """
    Predictions of the model on the unseen, 2020 season dataset.
    """
    )
    st.dataframe(data=preds_2020, width=None, height=None)

else:
    st.text("Unknown page selected.")
