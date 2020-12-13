import argparse
import pandas
from nba import br_extractor

__version__ = '0.1.0'

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    version = '%(prog)s ' + __version__
    parser.add_argument('--version', '-v', action='version', version=version)
    parser.add_argument("-e", "--extract", help="Extract data from BR", action="store_true")
    parser.add_argument("-t", "--train", help="Train model", action="store_true")
    parser.add_argument("-x", "--expose", help="Expose model on streamlit", action="store_true")
    return parser

def extract():
    extractor = br_extractor.BRExtractor()
    extract_player_stats(extractor, "./data/all_players_stats.csv")
    extract_mvp_votes(extractor, "./data/all_mvp_awards.csv")
    extract_teams_standings(extractor, "./data/all_teams_standings.csv")
    pass

def extract_player_stats(extractor, path):
    stats = extractor.get_player_stats()
    stats.to_csv(path)

def extract_mvp_votes(extractor, path):
    stats = extractor.get_mvp()
    stats.to_csv(path)

def extract_teams_standings(extractor, path):
    stats = extractor.get_team_standings()
    stats.to_csv(path)

def train():
    pass

def expose():
    pass

def main(args=None):
    """
    Main entry point for your project.
    Args:
        args : list
            A of arguments as if they were input in the command line. Leave it
            None to use sys.argv.
    """
    parser = get_parser()
    args = parser.parse_args(args)
    if args.extract: extract()
    if args.train: train()
    if args.expose: expose()

if __name__ == '__main__':
    main()