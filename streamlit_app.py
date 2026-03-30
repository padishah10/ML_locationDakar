from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from project_paths import (
    APPARTEMENTS_ML_CSV,
    APPARTEMENTS_MODEL,
    CHAMBRES_ML_CSV,
    CHAMBRES_MODEL,
    MAP_HTML,
    METRICS_JSON,
)
from train_modeles_loyer import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEGMENT_OPTIONS = {
    "Appartements": {
        "dataset_path": APPARTEMENTS_ML_CSV,
        "model_path": APPARTEMENTS_MODEL,
        "default_type": None,
    },
    "Chambres": {
        "dataset_path": CHAMBRES_ML_CSV,
        "model_path": CHAMBRES_MODEL,
        "default_type": "Chambre",
    },
}

st.set_page_config(page_title="Prediction des loyers a Dakar", layout="wide")


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_metrics(path: str):
    metrics_path = Path(path)
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def format_currency(value: float) -> str:
    return f"{int(round(value)):,} F CFA".replace(",", " ")


def build_prediction_input(df: pd.DataFrame, selected_segment: str) -> pd.DataFrame:
    quartiers = sorted(df["Quartier nettoye"].dropna().astype(str).unique().tolist())
    standings = sorted(df["Standing"].dropna().astype(str).unique().tolist())
    type_values = sorted(df["Type logement"].dropna().astype(str).unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        chambres = st.number_input("Nombre de chambres", min_value=1, max_value=15, value=1 if selected_segment == "Chambres" else 3)
        jours = st.number_input("Jours depuis publication", min_value=0, max_value=365, value=7)
        annee = st.number_input("Annee de publication", min_value=2024, max_value=2035, value=2026)
    with col2:
        mois = st.number_input("Mois de publication", min_value=1, max_value=12, value=3)
        jour = st.number_input("Jour de publication", min_value=1, max_value=31, value=27)
        standing = st.selectbox("Standing", standings, index=0 if standings else None)
    with col3:
        quartier = st.selectbox("Quartier", quartiers, index=0 if quartiers else None)
        if selected_segment == "Chambres":
            type_logement = "Chambre"
            st.text_input("Type logement", value=type_logement, disabled=True)
        else:
            default_type = "Appartement" if "Appartement" in type_values else (type_values[0] if type_values else "Appartement")
            type_logement = st.selectbox("Type logement", type_values, index=type_values.index(default_type) if type_values and default_type in type_values else 0)

    quartier_row = df[df["Quartier nettoye"] == quartier][["Latitude", "Longitude"]].dropna().head(1)
    latitude = float(quartier_row.iloc[0]["Latitude"]) if not quartier_row.empty else None
    longitude = float(quartier_row.iloc[0]["Longitude"]) if not quartier_row.empty else None

    return pd.DataFrame(
        [
            {
                "Chambres numeriques": chambres,
                "Jours depuis publication": jours,
                "Annee publication": annee,
                "Mois publication": mois,
                "Jour publication": jour,
                "Latitude": latitude,
                "Longitude": longitude,
                "Standing": standing,
                "Type logement": type_logement,
                "Quartier nettoye": quartier,
            }
        ]
    )


def render_overview(metrics, appartements_df: pd.DataFrame, chambres_df: pd.DataFrame) -> None:
    st.subheader("Vue d'ensemble")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Annonces appartements", len(appartements_df))
    kpi2.metric("Annonces chambres", len(chambres_df))
    kpi3.metric("Prix median appartement", format_currency(appartements_df["Prix numerique"].median()))
    kpi4.metric("Prix median chambre", format_currency(chambres_df["Prix numerique"].median()))

    if metrics:
        st.subheader("Metriques des modeles")
        metrics_df = pd.DataFrame(metrics).T[["segment", "lignes_total", "mae", "rmse", "r2", "prix_median"]]
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("Le fichier de metriques est absent. Lance `train_modeles_loyer.py` pour l'alimenter.")

    chart_df = pd.DataFrame(
        {
            "Segment": ["Appartements", "Chambres"],
            "Prix median": [appartements_df["Prix numerique"].median(), chambres_df["Prix numerique"].median()],
        }
    )
    st.bar_chart(chart_df.set_index("Segment"))


def render_prediction() -> None:
    st.subheader("Predire un loyer")
    selected_segment = st.radio("Segment", list(SEGMENT_OPTIONS.keys()), horizontal=True)
    segment_config = SEGMENT_OPTIONS[selected_segment]

    dataset_path = str(segment_config["dataset_path"])
    model_path = str(segment_config["model_path"])

    if not Path(model_path).exists():
        st.error(f"Modele introuvable: {model_path}. Lance `train_modeles_loyer.py`.")
        return

    df = load_dataset(dataset_path)
    model = load_model(model_path)

    prediction_input = build_prediction_input(df, selected_segment)
    st.dataframe(prediction_input, use_container_width=True)

    if st.button("Lancer la prediction", type="primary"):
        prediction = float(model.predict(prediction_input[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0])
        st.success(f"Loyer predit: {format_currency(prediction)}")

        comparable = df[df["Quartier nettoye"] == prediction_input.iloc[0]["Quartier nettoye"]]
        if not comparable.empty:
            st.caption("Comparaison locale")
            st.write(
                {
                    "Prix median quartier": format_currency(comparable["Prix numerique"].median()),
                    "Nombre d'annonces comparables": int(len(comparable)),
                }
            )


def render_map(df: pd.DataFrame) -> None:
    st.subheader("Carte des quartiers")
    if MAP_HTML.exists():
        components.html(MAP_HTML.read_text(encoding="utf-8"), height=600, scrolling=True)
    else:
        st.info("Le fichier de carte HTML est absent. Lance `affichage_carte.py`.")

    st.subheader("Points geographiques du dataset")
    points = df[["Latitude", "Longitude", "Prix numerique", "Quartier nettoye"]].dropna().copy()
    if points.empty:
        st.warning("Aucune coordonnee disponible.")
        return
    st.map(points.rename(columns={"Latitude": "lat", "Longitude": "lon"}))


def render_quality(df: pd.DataFrame) -> None:
    st.subheader("Qualite des donnees")
    missing_ratio = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    st.dataframe(missing_ratio.rename("Taux manquant (%)"), use_container_width=True)

    st.subheader("Apercu du dataset global")
    columns = [column for column in ["Categorie annonce", "Titre", "Quartier nettoye", "Prix numerique", "Standing", "Type logement"] if column in df.columns]
    st.dataframe(df[columns].head(50), use_container_width=True)


def main() -> None:
    st.title("Tableau de bord de prediction des loyers a Dakar")
    st.caption("Visualisation des donnees, metriques des modeles, prediction et cartographie")

    appartements_df = load_dataset(str(APPARTEMENTS_ML_CSV))
    chambres_df = load_dataset(str(CHAMBRES_ML_CSV))
    global_df = pd.concat([appartements_df.assign(Segment="Appartements"), chambres_df.assign(Segment="Chambres")], ignore_index=True)
    metrics = load_metrics(str(METRICS_JSON))

    tab1, tab2, tab3, tab4 = st.tabs(["Vue d'ensemble", "Prediction", "Carte", "Qualite des donnees"])

    with tab1:
        render_overview(metrics, appartements_df, chambres_df)
    with tab2:
        render_prediction()
    with tab3:
        render_map(global_df)
    with tab4:
        render_quality(global_df)


if __name__ == "__main__":
    main()

