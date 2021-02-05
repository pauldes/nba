import datetime
import os

import streamlit as st
import pandas
import joblib

from nba import br_extractor, preprocess, evaluate

# Constants
LOGO_URL = "https://i.jebbit.com/images/k9VpjZfZ/business-images/41hdlnbMRJSZe152NgYk_KIA_PerfAwards_MVP.png"
PAGE_PREDICTIONS = "Current year predictions"
PAGE_PERFORMANCE = "Model performance analysis"
CONFIDENCE_MODE_SOFTMAX = "Softmax-based"
CONFIDENCE_MODE_SHARE = "Percentage-based"

year = datetime.datetime.now().year
month = datetime.datetime.now().month
day = datetime.datetime.now().day

# Page properties
st.set_page_config(page_title='NBA MVP Prediction', page_icon = LOGO_URL, layout = 'centered', initial_sidebar_state = 'auto')

# Functions
#@st.cache
def load_model():
    return joblib.load('static/model/model.joblib')
@st.cache
def load_player_stats(day, month, season):
    extractor = br_extractor.BRExtractor()
    stats = extractor.get_player_stats(subset_by_seasons=[season], subset_by_stat_types=['per_game', 'per_36min', 'per_100poss', 'advanced'])
    stats.to_csv("./data/current/current_player_stats.csv")
    return stats
@st.cache
def load_team_stats(day, month, season):
    extractor = br_extractor.BRExtractor()
    stats = extractor.get_team_standings(subset_by_seasons=[season])
    stats.to_csv("./data/current/current_team_stats.csv")
    return stats
@st.cache
def consolidate_stats(team_stats, player_stats):
    player_stats["SEASON"] = player_stats["SEASON"].astype(int)
    team_stats["SEASON"] = team_stats["SEASON"].astype(int)
    stats = player_stats.merge(team_stats, how='inner', on=["TEAM", "SEASON"])
    stats = stats.set_index("PLAYER", drop=True)
    stats.to_csv("./data/current/current_consolidated_raw.csv")
    return stats
def load_2020_preds():
    preds = pandas.read_csv("./static/data/2020_dataset_predictions.csv")
    return preds
def load_test_preds():
    preds = pandas.read_csv("./static/data/test_dataset_predictions.csv")
    return preds
def mvp_found_pct(test_dataset_predictions):
    metrics = (test_dataset_predictions["Pred. MVP"] == test_dataset_predictions["True MVP"]).sum() / len(test_dataset_predictions)
    metrics = int(metrics*100)
    return str(metrics) + " %"
def avg_real_mvp_rank(test_dataset_predictions):
    metrics = (test_dataset_predictions["REAL_RANK"]).mean()
    return "%.2f" % metrics
def clean_data(data):
    #TODO : reuse cleaning process
    data = data.fillna(0.0)
    data["G"] = data["G"].astype(int)
    data = data[data["G"] >= int(0.4*data["G"].max())]
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
    series = series[series>filter_above]
    if filename not in os.listdir(folder):
        series.to_csv(folder + filename, header=False) 

def predict(data, model):
    # TODO get automatically from training step.. or keep all 
    cat = ['POS', 'CONF']
    # TODO get automatically from training step.. or keep all 
    num = ['2P%', '2PA_per_36min', '3P%', '3PAR_advanced', '3P_per_game', 'AGE', 'AST%_advanced', 'BLK_per_36min', 'DBPM_advanced', 'DRB_per_game', 'DRTG_per_100poss', 'DWS_advanced', 'FG%', 'FGA_per_game', 'FG_per_36min', 'FT%', 'FTA_per_100poss', 'FTR_advanced', 'G', 'MP', 'OBPM_advanced', 'ORB_per_36min', 'ORTG_per_100poss', 'PF_per_100poss', 'PF_per_game', 'STL_per_36min', 'TOV%_advanced', 'TOV_per_36min', 'TOV_per_game', 'TRB%_advanced', 'TS%_advanced', 'USG%_advanced', 'WS/48_advanced', 'WS_advanced', 'W', 'W/L%', 'GB', 'PL', 'PA/G', 'CONF_RANK']
    min_max_scaling = False
    data_processed_features_only, _ = preprocess.scale_per_value_of(data, cat, num, data["SEASON"], min_max_scaler=min_max_scaling)
    # TODO get automatically from training step
    features =  ['W', 'STL_per_36min', 'DRB_per_game', 'PL', 'TS%_advanced',
       'TOV%_advanced', 'FG%', 'FT%', 'GB', 'PF_per_100poss', '2P%', 'PA/G',
       '2PA_per_36min', '3P%', 'FG_per_36min', 'ORB_per_36min',
       'TRB%_advanced', 'TOV_per_game', 'OBPM_advanced', 'CONF_RANK', 'G',
       '3PAR_advanced', 'FTA_per_100poss', 'FGA_per_game', 'TOV_per_36min',
       'MP', 'AST%_advanced', 'AGE', 'PF_per_game', 'USG%_advanced',
       'DWS_advanced', 'ORTG_per_100poss', 'DBPM_advanced', 'WS/48_advanced',
       'FTR_advanced', 'DRTG_per_100poss', '3P_per_game', 'BLK_per_36min',
       'WS_advanced', 'W/L%', 'POS_C', 'POS_PF', 'POS_PG', 'POS_SF', 'POS_SG',
       'CONF_EASTERN_CONF', 'CONF_WESTERN_CONF']
    X = data_processed_features_only[features]
    preds = model.predict(X)
    return preds

# Init page
current_team_stats = load_team_stats(day, month, year).copy()
current_player_stats = load_player_stats(day, month, year).copy()
current_consolidated_raw = consolidate_stats(current_team_stats, current_player_stats)
preds_2020 = load_2020_preds()
preds_test = load_test_preds()
num_test_seasons = len(preds_test)
mvp_found_pct = mvp_found_pct(preds_test)
avg_real_mvp_rank = avg_real_mvp_rank(preds_test)
model = load_model()
dataset = clean_data(current_consolidated_raw)
# Predict
initial_columns = list(dataset.columns)
predictions = predict(dataset, model)
dataset.loc[:, "PRED"] = predictions
dataset = dataset.sort_values(by="PRED", ascending=False)
dataset.loc[:, "PRED_RANK"] = dataset["PRED"].rank(ascending=False)
save_predictions(dataset["PRED"], day, month, year)

# Sidebar
st.sidebar.image(LOGO_URL, width=100, clamp=False, channels='RGB', output_format='auto')
st.sidebar.text(f"Season : {year-1}-{year}")
st.sidebar.markdown(f'''
**Predicting the NBA Most Valuable Player using machine learning.**
''')
navigation_page = st.sidebar.radio('Navigate to', [PAGE_PREDICTIONS, PAGE_PERFORMANCE])
st.sidebar.markdown(f'''
*Made by [pauldes](https://github.com/pauldes). Code on [GitHub](https://github.com/pauldes/nba-mvp-prediction).*
''')

st.title(f'Predicting the NBA MVP.')

if navigation_page == PAGE_PREDICTIONS:

    st.header("Current year predictions")
    col1, col2 = st.beta_columns(2)
    col2.subheader("Prediction parameters")
    confidence_mode = col2.radio('MVP probability estimation method', [CONFIDENCE_MODE_SHARE, CONFIDENCE_MODE_SOFTMAX])
    compute_probs_based_on_top_n = col2.slider('Number of players used to estimate probability', min_value=5, max_value=100, value=10, step=5)
    if confidence_mode == CONFIDENCE_MODE_SOFTMAX:
        dataset.loc[dataset.PRED_RANK <= compute_probs_based_on_top_n, "MVP probability"] = evaluate.softmax(dataset[dataset.PRED_RANK <= compute_probs_based_on_top_n]["PRED"]) * 100
    else:
        dataset.loc[dataset.PRED_RANK <= compute_probs_based_on_top_n, "MVP probability"] = evaluate.share(dataset[dataset.PRED_RANK <= compute_probs_based_on_top_n]["PRED"]) * 100
    dataset.loc[dataset.PRED_RANK > compute_probs_based_on_top_n, "MVP probability"] = 0.
    dataset["MVP probability"] = dataset["MVP probability"].map('{:,.2f}%'.format)
    dataset["MVP rank"] = dataset["PRED_RANK"]
    show_columns = ["MVP probability", "MVP rank"] + initial_columns[:]
    dataset = dataset[show_columns]
    st.subheader(f"Predicted top {compute_probs_based_on_top_n} MVP ranking")
    st.dataframe(data=dataset.head(compute_probs_based_on_top_n), width=None, height=None)

    top_3 = dataset["MVP probability"][:3].to_dict()

    for n, player_name in enumerate(top_3):
        title_level = "###" + n*"#"
        col1.markdown(f'''
        {title_level} #{n+1} : **{player_name}** 
        
        *{top_3[player_name]} chance to win MVP*
        ''')

elif navigation_page == PAGE_PERFORMANCE:

    st.header("Model performance analysis")
    st.subheader(f"Test predictions ({num_test_seasons} seasons)")
    st.markdown('''
    Predictions of the model on the unseen, test dataset.
    ''')
    st.markdown(f'''
    - **{mvp_found_pct}** of MVPs correctly found
    - Real MVP is ranked in average **{avg_real_mvp_rank}**
    ''')
    st.dataframe(data=preds_test, width=None, height=None)
    st.subheader("Year 2020")
    st.markdown('''
    Predictions of the model on the unseen, 2020 season dataset.
    ''')
    st.dataframe(data=preds_2020, width=None, height=None)

else:
    st.text("Unknown page selected.")
