# Predicting the NBA Most Valuable Player

Look for the result at [nbamvp.herokuapp.com](https://nbamvp.herokuapp.com)

![](static/img/animated_screenshot.gif)

## Development

Run the webapp locally : 
```pipenv run streamlit run app.py```

Load fresh data :
```pipenv run python main.py --help```

Track models performance :
```pipenv run mlflow ui```

## Main challenges


**Imbalanced data** : there is only 1 MVP per year, among hundreds of players.

Solutions :
- use MVP share instead of MVP award
- use generally accepted tresholds to filter non-MVP players : more than 40% of games played

**Label consistency** : a player winning MVP one year may not have won MVP the year before, event with the same stats. It all depends on the other players competition.

Solutions :
- normalize stats per season
  - min-max scaling
  - standardization

## Future work

- Rank stats (another solution for label consistency issue)
- Use previous years result (to model voters lassitude phenomena)
- Limit the players pool in each team to 2 or 3 players based on a treshold to define (or on another model)
