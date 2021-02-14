Predicting the KIA MVP

pipenv run mlflow ui

pipenv run streamlit run app.py

pipenv run python main.py --help

Probleme anticipé :
- 1 vice MVP une année pourrait être MVP l'année d'avant. La concurrence est importante > normalisation, mais ne suffit pas...
- Win Share semble prendre trop de poids. Essayer de l'enlever.
- log transform de la cible pertinent ?
- softmax ou percentage du share lors du training ?
- Joker is predicted first with a low confidence rank (7)... It is not normal


- Building SHAP charts is very slow. Should be built in batch every day.