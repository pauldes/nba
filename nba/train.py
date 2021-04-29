import random
import pandas
import seaborn
import numpy
from matplotlib import pyplot
from qbstyles import mpl_style
from sklearn import (
    dummy,
    tree,
    model_selection,
    metrics,
    preprocessing,
    linear_model,
    ensemble,
    neural_network,
)
import importlib
import mlflow
import joblib

from nba import preprocess, analyze, evaluate

data = pandas.read_csv("../data/training/all_consolidated_final.csv")
data = data.set_index("player_season_team", drop=True)
print(data.sample(3))

print(
    "Before filters: ",
    len(data),
    "players -",
    len(data[data.MVP_CANDIDATE]),
    "MVP candidates -",
    len(data[data.MVP_WINNER]),
    "winners",
)

data_copy = data.copy()
# Apply filters
data = data[data["G"] >= int(0.4 * data["G"].max())]  # TODO use season nb of game
data = data[data["CONF_RANK"] <= 8]
data = data[data["MP"] >= 20.0]

removed_players = data_copy.loc[~data_copy.index.isin(data.index)]
cols = ["SEASON", "G", "CONF_RANK", "MP"]
removed_mvp_candidates = removed_players[removed_players.MVP_CANDIDATE]
print(len(removed_mvp_candidates), "MVP candidates removed due to filters")
if len(removed_mvp_candidates) > 0:
    print(removed_mvp_candidates.sample(5)[cols])

print(
    "After filters: ",
    len(data),
    "players -",
    len(data[data.MVP_CANDIDATE]),
    "MVP candidates -",
    len(data[data.MVP_WINNER]),
    "winners",
)

raise NotImplementedError("Run train_model.ipynb manually")
