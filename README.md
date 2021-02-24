Predicting the KIA MVP

Look for the result at [nbamvp.herokuapp.com](https://nbamvp.herokuapp.com)

```pipenv run mlflow ui```

```pipenv run streamlit run app.py```

```pipenv run python main.py --help```

Probleme anticipé :
- 1 vice MVP une année pourrait être MVP l'année d'avant. La concurrence est importante. C'est cela que vise à traiter la normalisation par saison. Ajouter un rank pour chaque stat pourrait aussi être pertinent.