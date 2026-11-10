# Loyers à Dakar

Estimation des loyers d'appartements à Dakar à partir d'annonces publiées en ligne.
Projet de mémoire de Master 2 MIAGE.

## Contenu du dépôt

| Élément | Rôle |
|---|---|
| `streamlit_app.py` | Application (point d'entrée Streamlit Community Cloud) |
| `requirements.txt` | Dépendances de l'application |
| `.streamlit/config.toml` | Couleurs et thème de l'application |
| `donnees/` | Annonces brutes, fichiers intermédiaires et jeux de données finaux |
| `pipeline/` | Scripts de collecte, de nettoyage, d'entraînement et d'évaluation |
| `resultats/` | Métriques et tableaux d'évaluation du modèle |

L'application entraîne son modèle au démarrage à partir de
`donnees/appartements_dakar_ml.csv` : aucun fichier de modèle n'est versionné.

## Lancer l'application

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Relancer la chaîne de traitement

```powershell
pip install -r pipeline/requirements.txt
cd pipeline
python lancer_pipeline.py
```

Les étapes, dans l'ordre : extraction des attributs (`nlp_extraction.py`),
normalisation des quartiers (`quartierNettoye.py`), géocodage (`quartiers_dakar.py`),
filtrage et séparation (`separer_datasets.py`), entraînement (`train_modeles_loyer.py`),
évaluation (`evaluer_modeles.py`).

Pour une nouvelle collecte, lancer d'abord `python webscraping.py` : les nouvelles
annonces s'ajoutent à `donnees/annonces_dakar.csv`.

Les annonces écartées (locations à la journée, bureaux, ventes, doublons, prix absents)
sont listées avec leur motif dans `donnees/annonces_exclues.csv`.
