import argparse
import datetime
import pandas
from nba import br_extractor, clean, train

__version__ = "0.1.0"
year = datetime.datetime.now().year


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    version = "%(prog)s " + __version__
    parser.add_argument("--version", "-v", action="version", version=version)
    parser.add_argument(
        "-e", "--extract", help="Extract data from BR", action="store_true"
    )
    parser.add_argument(
        "-c",
        "--consolidate",
        help="Consolidate extracted data to build dataset",
        action="store_true",
    )
    parser.add_argument("-t", "--train", help="Train model", action="store_true")
    return parser


def extract():
    extractor = br_extractor.BRExtractor()
    extract_player_stats(extractor, "./data/training/all_players_stats.csv")
    extract_mvp_votes(extractor, "./data/training/all_mvp_votes.csv")
    extract_teams_standings(extractor, "./data/training/all_teams_standings.csv")


def extract_player_stats(extractor, path):
    # We do not retrieve totals stats since we want to be able to predict at any moment in the season
    # That's not a big deal since we will have total games played, stats per game and per minute (will be highly correlated)
    # We could have normalized totals within the season if we'd have really want to use them
    stats = extractor.get_player_stats(
        subset_by_stat_types=["per_game", "per_36min", "per_100poss", "advanced"]
    )
    stats.to_csv(path)


def extract_mvp_votes(extractor, path):
    stats = extractor.get_mvp()
    stats.to_csv(path)


def extract_teams_standings(extractor, path):
    stats = extractor.get_team_standings()
    stats.to_csv(path)


def consolidate():
    standings = pandas.read_csv("./data/training/all_teams_standings.csv")
    awards = pandas.read_csv("./data/training/all_mvp_votes.csv")
    awards_col = ["MVP_VOTES_SHARE", "MVP_WINNER", "MVP_PODIUM", "MVP_CANDIDATE"]
    awards = awards[["player_season_team"] + awards_col]
    stats = pandas.read_csv("./data/training/all_players_stats.csv")
    all_data = stats.merge(awards, how="left", on="player_season_team")
    for col in "MVP_WINNER", "MVP_PODIUM", "MVP_CANDIDATE":
        all_data.loc[all_data["SEASON"] != year, col] = all_data[col].fillna(False)
    all_data.loc[all_data["SEASON"] != year, "MVP_VOTES_SHARE"] = all_data[
        "MVP_VOTES_SHARE"
    ].fillna(0.0)
    all_data = all_data.merge(standings, how="inner", on=["TEAM", "SEASON"])
    sample_cols = ["player_season_team", "MVP_VOTES_SHARE", "MVP_WINNER"]
    print(
        "Sample of MVP winners : \n",
        all_data[all_data["MVP_WINNER"] == True].sample(10)[sample_cols],
    )
    print(
        "Sample of MVP candidates : \n",
        all_data[all_data["MVP_CANDIDATE"] == True].sample(10)[sample_cols],
    )
    print(
        "Sample of MVP podium : \n",
        all_data[all_data["MVP_PODIUM"] == True].sample(10)[sample_cols],
    )
    print("MVPs : ", all_data[all_data["MVP_WINNER"] == True]["SEASON"].nunique())
    all_data = all_data.set_index("player_season_team", drop=True)
    path = "./data/training/all_consolidated_raw.csv"
    all_data.to_csv(path)


def retrain():
    clean()
    train()


def main(args=None):
    """Main entry point.

    Args:
        args : list of arguments as if they were input in the command line.
    """
    parser = get_parser()
    args = parser.parse_args(args)
    if args.extract:
        extract()
    if args.consolidate:
        consolidate()
    if args.train:
        retrain()

if __name__ == "__main__":
    main()
