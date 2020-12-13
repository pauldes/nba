import os
import pandas
import yaml
import pandas as pd
import requests
from bs4 import BeautifulSoup
from basketball_reference_scraper.teams import get_roster, get_team_stats
from basketball_reference_scraper.seasons import get_standings

""" 
1955-56 through 1979-1980: Voting was done by players. Rules prohibited player from voting for himself or any teammate.
1980-81 to present: Voting conducted by media. 
"""

class BRExtractor():

    default_team_names_file_path = "./nba/team_names.yaml"

    def __init__(self, team_names_file_path=default_team_names_file_path):
        self.team_names = self._get_yaml(team_names_file_path)

    @staticmethod
    def _get_yaml(yaml_file_path):
        """ Get a YAML file as a dictionnary.
        """
        with open(yaml_file_path, 'r') as stream:
            try:
                loaded = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("Could not load YAML file", yaml_file_path, ":", e)
                return dict()
            else:
                return loaded
    
    @staticmethod
    def retrieve_mvp_votes(season):
        season = str(season)
        root_url = "https://www.basketball-reference.com/"
        url = f"{root_url}awards/awards_{season}.html"
        r = requests.get(url)
        if r.status_code==200:
            soup = BeautifulSoup(r.content, 'html.parser')
            table_mvp = soup.find('table', id='mvp')
            table_nba_mvp = soup.find('table', id='nba_mvp')
            if table_mvp is not None:
                table = table_mvp
            elif table_nba_mvp is not None:
                table = table_nba_mvp
            else:
                raise Exception("No table found for MVP data for season", season)
            df = pd.read_html(str(table), header=1)[0]
            df.columns = [str(col).upper() for col in df.columns]
            df.loc[:, 'SEASON'] = season
            df = df.rename(columns={"SHARE":"MVP_VOTES_SHARE"})
            df = df.rename(columns={"TM":"TEAM"})
            df = df[["PLAYER", "TEAM", "SEASON", "MVP_VOTES_SHARE", "RANK"]]
            df.loc[:, 'PLAYER'] = df["PLAYER"].str.replace('[^A-Za-z]', '')
            df.loc[:, 'MVP_WINNER'] = False
            df["RANK"] = df["RANK"].astype(str).str.replace('[^0-9]', '').astype(int, errors='raise')
            df.loc[df["RANK"]==1, 'MVP_WINNER'] = True
            df.loc[:, 'MVP_PODIUM'] = False
            df.loc[df["RANK"].isin([1,2,3]), 'MVP_PODIUM'] = True
            df.loc[:, 'MVP_CANDIDATE'] = True
            df = df.drop("RANK", axis='columns')
            return df
        else:
            raise ConnectionError("Could not connect to BR and get data, status code : %s", r.status_code)

    def get_mvp(self, subset_by_seasons: list = None):
        allowed_seasons = range(1974, 2020)
        if subset_by_seasons is not None:
            seasons = [season for season in subset_by_seasons if season in allowed_seasons]
        else:
            seasons = allowed_seasons
        total_dfs = []
        for season in seasons:
            print("Retrieving MVP of season", season, "...")
            results = self.retrieve_mvp_votes(season)
            results.loc[:, "player_season_team"] = results["PLAYER"].str.replace(" ", "") + "_" + results["SEASON"] + "_" + results["TEAM"]
            results = results.set_index("player_season_team", drop=True)
            total_dfs.append(results)
        all_df = pandas.concat(total_dfs, join='outer', axis="index", ignore_index=False)
        return all_df


    @staticmethod
    def get_roster_stats_v2(season, stat_type):
        """
        Return all players stats for one season.
        Season : end year of season, as int or str
        Available stat types (case insensitive) : 'totals', 'per_game', 'per_36min', 'per_100poss', 'advanced'
        """
        root_url = "https://www.basketball-reference.com/"
        season = str(season)
        stat_type = str(stat_type).lower()
        url_mapper = {'totals':'totals', 'per_game':'per_game', 'per_36min':'per_minute', 'per_100poss':'per_poss', 'advanced':'advanced'}
        stat_type = url_mapper[stat_type]
        url = f"{root_url}leagues/NBA_{season}_{stat_type}.html"
        r = requests.get(url)
        if r.status_code==200:
                soup = BeautifulSoup(r.content, 'html.parser')
                table = soup.find('table')
                df = pd.read_html(str(table))[0]
                df = df.loc[df.Player!="Player",:]
                df.columns = [str(col).upper() for col in df.columns]
                df.loc[:, 'SEASON'] = season
                df.loc[:, 'PLAYER'] = df["PLAYER"].str.replace('[^A-Za-z]', '')
                df = df.rename(columns={"TM":"TEAM"})
                df = df.drop("RK", axis='columns')
                for col in df.columns:
                    if col.startswith("3P"):
                        df[col] = df[col].fillna(0.0)
                return df
        else:
            raise ConnectionError("Could not connect to BR and get data, status code : %s", r.status_code)

    def get_team_standings(self, subset_by_seasons: list = None):
        """ Assumptions : the season is over by June 1st.
        TODO : Use the season dataset to find last game date.
        """
        allowed_seasons = range(1974, 2021)
        if subset_by_seasons is not None:
            seasons = [season for season in subset_by_seasons if season in allowed_seasons]
        else:
            seasons = allowed_seasons
        total_dfs = []
        for season in seasons:
            print("Retrieving standings of season", season, "...")
            date = "06-01-" + str(season)
            dfs = []
            results = get_standings(date=date)
            for conference, df in results.items():
                df = df.dropna(axis='index', how='any')
                df = df.sort_values(by="W/L%", ascending=False)
                df = df.reset_index(drop=True)
                df.loc[:, "CONF"] = conference
                df.loc[:, "CONF_RANK"] = df.index + 1
                df.loc[:, "TEAM"] = df["TEAM"].str.upper().str.replace('[^A-Z]', '')
                team_names = {}
                for raw, short in self.team_names.items():
                    raw = ''.join(filter(str.isalpha, raw)).upper()
                    team_names[raw] = short
                df = df[~df["TEAM"].str.contains("DIVISION")]
                unmapped_teams = [team for team in df["TEAM"].unique() if team not in team_names.keys()]
                mapped_teams = [team for team in df["TEAM"].unique() if team in team_names.keys()]
                df.loc[:, "TEAM"] = df["TEAM"].map(team_names)
                if df["TEAM"].isna().sum() > 0:
                    raise ValueError("Unknown/unmapped teams : %s", unmapped_teams)
                df.loc[:, "GB"] = df["GB"].str.replace("—", "0.0").astype(float, errors="raise")
                df.loc[:, "TEAM_SEASON"] = df["TEAM"] + "_" + str(season)
                df.loc[:, "SEASON"] = season
                df = df.set_index("TEAM_SEASON", drop=True)
                dfs.append(df)
            all_conf_df = pandas.concat(dfs, join='outer', axis="index", ignore_index=False)
            total_dfs.append(all_conf_df)
        all_conf_df = pandas.concat(total_dfs, join='outer', axis="index", ignore_index=False)
        return all_conf_df

    def get_player_stats(self, subset_by_teams: list = None, subset_by_seasons: list = None, subset_by_stat_types: list = None):
        """
        Get a set of stats.
        Defaults to all teams, all seasons, all stat types.
        """

        allowed_stat_types = ['totals', 'per_game', 'per_36min', 'per_100poss', 'advanced']
        allowed_seasons = range(1974, 2021)
        allowed_teams = list(set(self.team_names.values()))

        if subset_by_teams is not None:
            subset_by_teams = [str(s).upper() for s in subset_by_teams]

        if subset_by_seasons is not None:
            seasons = [season for season in subset_by_seasons if season in allowed_seasons]
        else:
            seasons = allowed_seasons
        if subset_by_stat_types is not None:
            subset_by_stat_types = [str(s).lower() for s in subset_by_stat_types]
            stat_types = [stat_type for stat_type in subset_by_stat_types if stat_type in allowed_stat_types]
        else:
            stat_types = allowed_stat_types
        
        season_dfs = []
        for season in seasons:
            do_not_suffix = ["PLAYER", "POS", "AGE", "TEAM", "SEASON", "G", "GS", "FG%", "3P%", "FT%", "2P%", "eFG%"]
            stat_type_dfs = []
            for stat_type in stat_types:
                print("Retrieving", stat_type, "stats for season", season,"...")
                try:
                    stat_type_df = self.get_roster_stats_v2(season, stat_type)
                except Exception as e:
                    print("Could not retrieve data. Are you sure", team,"played in season", str(season), "?", e)
                else:
                    stat_type_df.columns = [col + "_" + str(stat_type) if col not in do_not_suffix else col for col in stat_type_df.columns]
                    stat_type_df.loc[:, "player_season_team"] = stat_type_df["PLAYER"].str.replace(" ", "") + "_" + stat_type_df["SEASON"] + "_" + stat_type_df["TEAM"]
                    stat_type_df = stat_type_df.set_index("player_season_team", drop=True)
                    stat_type_df = stat_type_df.dropna(axis="columns", how="all")
                    stat_type_dfs.append(stat_type_df)
                    
            season_df = pandas.concat(stat_type_dfs, join='outer', axis="columns", ignore_index=False)
            season_df = season_df.loc[:, ~season_df.columns.duplicated()]
            season_dfs.append(season_df)

        full_df = pandas.concat(season_dfs, join='outer', axis="index", ignore_index=False)

        if subset_by_teams is not None:
            full_df = full_df[full_df.TEAM.isin(subset_by_teams)]

        return full_df