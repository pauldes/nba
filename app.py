import datetime

import streamlit as st
import pandas

from nba import br_extractor

# Constants
logo_url = "https://i.jebbit.com/images/k9VpjZfZ/business-images/41hdlnbMRJSZe152NgYk_KIA_PerfAwards_MVP.png"
year = datetime.datetime.now().year

# Page properties
st.set_page_config(page_title='NBA MVP Prediction', page_icon = logo_url, layout = 'centered', initial_sidebar_state = 'auto')
st.title(f'Predicting the MVP.')

# Functions
@st.cache
def load_player_stats(season):
    extractor = br_extractor.BRExtractor()
    stats = extractor.get_player_stats(subset_by_seasons=[season], subset_by_stat_types=['per_game', 'per_36min', 'per_100poss', 'advanced'])
    stats.to_csv("./data/current_player_stats.csv")
    return stats
@st.cache
def load_team_stats(season):
    extractor = br_extractor.BRExtractor()
    stats = extractor.get_team_standings(subset_by_seasons=[season])
    stats.to_csv("./data/current_team_stats.csv")
    return stats
@st.cache
def consolidate_stats(team_stats, player_stats):
    stats = player_stats.merge(team_stats, how='inner', on=["TEAM"])
    #stats = stats.set_index("player_season_team", drop=True)
    stats.to_csv("./data/current_consolidated_raw.csv")
    return stats
def load_2020_preds():
    preds = pandas.read_csv("./static/data/2020_dataset_predictions.csv")
    return preds
def load_test_preds():
    preds = pandas.read_csv("./static/data/test_dataset_predictions.csv")
    return preds

# Init page
current_team_stats = load_team_stats(year)
current_player_stats = load_player_stats(year)
current_consolidated_raw = consolidate_stats(current_team_stats, current_player_stats)
preds_2020 = load_2020_preds()
preds_test = load_test_preds()

# Sidebar
st.sidebar.image(logo_url, width=100, clamp=False, channels='RGB', output_format='auto')
st.sidebar.text(f"Season : {year-1}-{year}")
st.sidebar.markdown('''
**Predicting the NBA Most Valuable Player using machine learning.**

Model used :

Performance :
- Training set : XX%
- Validation set : YY%
- Test set : ZZ%


Made by [pauldes](https://github.com/pauldes/nba-mvp-prediction).
''')

# Main content
st.header("Data retrieved")
st.subheader("Player stats")
st.markdown('''
These stats describe the player individual accomplishments.
''')
st.dataframe(data=current_player_stats.sample(10), width=None, height=None)
st.subheader("Team stats")
st.markdown('''
These stats describe the team accomplishments.
''')
st.dataframe(data=current_team_stats.sample(10), width=None, height=None)
st.header("Model performance")
st.subheader("Test dataset")
st.markdown('''
Predictions of the model on the unseen, test dataset.
''')
st.dataframe(data=preds_2020, width=None, height=None)
st.subheader("Year 2020")
st.markdown('''
Predictions of the model on the unseen, 2020 season dataset.
''')
st.dataframe(data=preds_test, width=None, height=None)
st.header("Current year predictions")