# Predicting the NBA Most Valuable Player

This project aims at predicting the player who will win the NBA MVP award, by modelling the voting panel. The National Basketball Association (NBA) is given since the 1955–56 season to the best performing player of the regular season. Until the 1979–80 season, the MVP was selected by a vote of NBA players. Since the 1980–81 season, the award is decided by a panel of sportswriters and broadcasters - more info [here](https://en.wikipedia.org/wiki/NBA_Most_Valuable_Player_Award).

Have a look at the last model predictions at [nbamvp.herokuapp.com](https://nbamvp.herokuapp.com) !

![](static/img/animated_screenshot.gif)

## Development

Run the webapp locally : 
```pipenv run streamlit run app.py```

Load fresh data :
```pipenv run python main.py --help```

Track models performance :
```pipenv run mlflow ui```

## Main challenges


#### Imbalanced data 

There is only 1 MVP per year, among hundreds of players.

Solutions :
- use MVP share instead of MVP award
- use generally accepted tresholds to filter non-MVP players : more than 40% of games played

#### Label consistency

A player winning MVP one year may not have won MVP the year before, event with the same stats. It all depends on the other players competition.

Solutions :
- normalize stats per season
  - min-max scaling
  - standardization

## Future work and model improvement ideas

- Rank stats (another solution for label consistency issue)
- Use previous years result (to model voters lassitude phenomena)
- Limit the players pool in each team to 2 or 3 players based on a treshold to define (or on another model)
