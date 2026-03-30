# Application Streamlit - Prediction des loyers a Dakar

## Contenu du depot
- `streamlit_app.py` : point d'entree Streamlit Community Cloud
- `project_paths.py` : chemins relatifs du projet
- `requirements.txt` : dependances Python
- `Fichiers CSV/` : datasets affiches par l'application
- `Modeles/` : modeles entraines charges a l'execution
- `Metriques/` : metriques affichees dans le tableau de bord

## Deploiement sur Streamlit Community Cloud
1. Creer un depot GitHub et y pousser tout le contenu de ce dossier.
2. Dans Streamlit Community Cloud, choisir ce depot.
3. Definir le fichier principal sur `streamlit_app.py`.
4. Choisir une version Python compatible dans les parametres avances si besoin.

## Execution locale
```powershell
python -m streamlit run streamlit_app.py
```
