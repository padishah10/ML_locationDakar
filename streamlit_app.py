"""Estimateur de loyers a Dakar - application de demonstration.

Le modele est entraine au demarrage a partir du fichier d'annonces
(donnees/appartements_dakar_ml.csv) : aucun fichier de modele a livrer, et
aucun probleme de version entre la machine de developpement et le serveur.
"""
from __future__ import annotations

import re
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent / "donnees" / "appartements_dakar_ml.csv"

NUMERIC_FEATURES = ["Chambres numeriques", "Latitude", "Longitude"]
CATEGORICAL_FEATURES = ["Standing", "Type logement", "Quartier nettoye"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Prix numerique"
RANDOM_STATE = 42

MIN_ANNONCES_QUARTIER_FIABLE = 10   # en dessous : estimation signalee comme indicative
MIN_ANNONCES_GRAPHIQUE = 15         # quartiers affiches dans le classement
MIN_ANNONCES_CARTE = 3

COULEUR = "#0B5563"
RAMPE = ["#CFE3E2", "#93C1C1", "#559C9F", "#1F7480", "#0B4250"]  # une seule teinte, du clair au fonce
PALIERS = [300_000, 500_000, 800_000, 1_200_000]

# Type de bien : libelle affiche -> (valeur du modele, nombre de chambres impose ou None)
TYPES_DE_BIEN = {
    "Appartement": ("Appartement", None),
    "Studio": ("Studio", 1),
    "F2 · 1 chambre et salon": ("F2", 1),
    "F3 · 2 chambres et salon": ("F3", 2),
    "F4 · 3 chambres et salon": ("F4", 3),
    "F5 · 4 chambres et salon": ("F5", 4),
    "Duplex": ("Duplex", None),
    "Penthouse": ("Penthouse", None),
    "Villa": ("Villa", None),
}
STANDINGS = {"Standard": "Standard", "Meublé": "Meuble", "Haut standing": "Luxe"}
TYPE_LISIBLE = {"Appartement": "Appartement", "Meuble": "Meublé", "Luxe": "Haut standing"}

ACCENTS_QUARTIER = [
    (r"\bCite\b", "Cité"),
    (r"\bLiberte\b", "Liberté"),
    (r"\bAmitie\b", "Amitié"),
    (r"\bGuediawaye\b", "Guédiawaye"),
    (r"\bDerkle\b", "Derklé"),
    (r"\bHlm\b", "HLM"),
    (r"\bZone De Captage\b", "Zone de Captage"),
    (r"\bPatte D.Oie\b", "Patte d'Oie"),
]


# --------------------------------------------------------------------------- mise en forme
def fcfa(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ") + " F CFA"


def arrondi(value: float, pas: int = 5_000) -> int:
    return int(round(value / pas) * pas)


def nom_quartier(value: str) -> str:
    text = str(value)
    for pattern, replacement in ACCENTS_QUARTIER:
        text = re.sub(pattern, replacement, text)
    return "Dakar (quartier non précisé)" if text == "Dakar" else text


# --------------------------------------------------------------------------- donnees et modele
@st.cache_data(show_spinner=False)
def charger_annonces() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")
    df = df.dropna(subset=[TARGET, "Quartier nettoye"]).copy()
    df[TARGET] = df[TARGET].astype(float)
    df["Quartier"] = df["Quartier nettoye"].map(nom_quartier)
    return df


def construire_modele() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    regressor = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)),
        ]
    )


@st.cache_resource(show_spinner="Préparation de l'estimateur…")
def entrainer():
    """Entraine le modele et mesure sa fiabilite sur des annonces qu'il n'a pas vues."""
    df = charger_annonces()
    x, y = df[FEATURES], df[TARGET]

    # Chaque annonce est estimee par un modele qui ne l'a pas vue (5 groupes).
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    estimations = cross_val_predict(construire_modele(), x, y, cv=folds)
    rapport = y.to_numpy() / estimations
    ecart = np.abs(estimations - y.to_numpy()) / y.to_numpy()
    fiabilite = {
        # Fourchette contenant 2 loyers reels sur 3 autour de l'estimation.
        "bas": float(np.quantile(rapport, 1 / 6)),
        "haut": float(np.quantile(rapport, 5 / 6)),
        "ecart_median": float(np.median(ecart)),
        "part_20": float((ecart <= 0.20).mean()),
        "part_30": float((ecart <= 0.30).mean()),
    }
    modele = construire_modele().fit(x, y)
    return modele, fiabilite


@st.cache_data(show_spinner=False)
def resume_quartiers() -> pd.DataFrame:
    df = charger_annonces()
    resume = (
        df.groupby("Quartier nettoye")
        .agg(
            Quartier=("Quartier", "first"),
            annonces=(TARGET, "size"),
            loyer_median=(TARGET, "median"),
            loyer_bas=(TARGET, lambda s: s.quantile(0.25)),
            loyer_haut=(TARGET, lambda s: s.quantile(0.75)),
            Latitude=("Latitude", "first"),
            Longitude=("Longitude", "first"),
        )
        .reset_index()
    )
    return resume.sort_values("annonces", ascending=False)


# --------------------------------------------------------------------------- habillage
def habillage() -> None:
    st.markdown(
        f"""
        <style>
          #MainMenu, footer, header [data-testid="stToolbar"] {{visibility: hidden;}}
          .block-container {{padding-top: 2.2rem; max-width: 1150px;}}
          .entete h1 {{font-size: 2.1rem; font-weight: 700; color: {COULEUR}; margin: 0; letter-spacing: -0.01em;}}
          .entete p {{color: #52606D; font-size: 1.05rem; margin: .35rem 0 1.2rem 0;}}
          .carte-resultat {{background: #FFFFFF; border: 1px solid #DDE5E3; border-left: 6px solid {COULEUR};
                           border-radius: 10px; padding: 1.4rem 1.6rem; margin: .4rem 0 1rem 0;}}
          .carte-resultat .libelle {{color: #52606D; font-size: .82rem; text-transform: uppercase; letter-spacing: .06em;}}
          .carte-resultat .loyer {{font-size: 2.6rem; font-weight: 700; color: #1F2933; line-height: 1.15; margin: .15rem 0;}}
          .carte-resultat .par-mois {{font-size: 1rem; font-weight: 400; color: #52606D;}}
          .carte-resultat .fourchette {{font-size: 1.05rem; color: #1F2933; margin-top: .5rem;}}
          .carte-resultat .note {{color: #52606D; font-size: .9rem; margin-top: .35rem;}}
          .legende {{display: flex; flex-wrap: wrap; gap: 1.1rem; color: #52606D; font-size: .85rem; margin: .2rem 0 1rem 0;}}
          .legende span {{display: inline-flex; align-items: center; gap: .4rem;}}
          .legende i {{width: 14px; height: 14px; border-radius: 50%; display: inline-block;}}
          [data-testid="stMetricValue"] {{color: #1F2933;}}
        </style>
        <div class="entete">
          <h1>Loyers à Dakar</h1>
          <p>Estimez le loyer mensuel d'un appartement à partir des annonces réellement publiées sur le marché.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- onglet 1 : estimation
def onglet_estimation(df: pd.DataFrame, modele: Pipeline, fiabilite: dict) -> None:
    quartiers = resume_quartiers()
    libelles = dict(zip(quartiers["Quartier"], quartiers["Quartier nettoye"]))

    col1, col2, col3, col4 = st.columns([1.3, 1.3, 1, 1])
    with col1:
        quartier_affiche = st.selectbox("Quartier", sorted(libelles), index=sorted(libelles).index("Almadies"))
    with col2:
        type_affiche = st.selectbox("Type de bien", list(TYPES_DE_BIEN), index=list(TYPES_DE_BIEN).index("F4 · 3 chambres et salon"))
    type_modele, chambres_imposees = TYPES_DE_BIEN[type_affiche]
    with col3:
        if chambres_imposees is None:
            chambres = st.number_input("Chambres", min_value=1, max_value=8, value=3, step=1)
        else:
            chambres = chambres_imposees
            st.text_input("Chambres", value=str(chambres), disabled=True)
    with col4:
        if type_modele == "Penthouse":
            # Un penthouse est par nature un bien haut de gamme.
            standing_affiche = "Haut standing"
            st.text_input("Standing", value=standing_affiche, disabled=True)
        else:
            standing_affiche = st.selectbox("Standing", list(STANDINGS))

    quartier = libelles[quartier_affiche]
    infos = quartiers[quartiers["Quartier nettoye"] == quartier].iloc[0]
    bien = pd.DataFrame(
        [
            {
                "Chambres numeriques": chambres,
                "Latitude": infos["Latitude"],
                "Longitude": infos["Longitude"],
                "Standing": STANDINGS[standing_affiche],
                "Type logement": type_modele,
                "Quartier nettoye": quartier,
            }
        ]
    )
    estimation = float(modele.predict(bien[FEATURES])[0])
    bas, haut = arrondi(estimation * fiabilite["bas"]), arrondi(estimation * fiabilite["haut"])

    st.markdown(
        f"""
        <div class="carte-resultat">
          <div class="libelle">Loyer estimé · {type_affiche.split(' · ')[0]}, {chambres} chambre{'s' if chambres > 1 else ''}, {standing_affiche.lower()} · {quartier_affiche}</div>
          <div class="loyer">{fcfa(arrondi(estimation))} <span class="par-mois">par mois</span></div>
          <div class="fourchette">Fourchette probable : <b>{fcfa(bas)}</b> à <b>{fcfa(haut)}</b></div>
          <div class="note">Deux loyers réels sur trois se situent dans cette fourchette pour un bien de ce profil.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    annonces_quartier = int(infos["annonces"])
    if annonces_quartier < MIN_ANNONCES_QUARTIER_FIABLE:
        st.warning(
            f"Seulement {annonces_quartier} annonce{'s' if annonces_quartier > 1 else ''} à {quartier_affiche} : "
            "l'estimation s'appuie surtout sur les quartiers voisins et reste indicative."
        )

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Loyer médian à {quartier_affiche}", fcfa(infos["loyer_median"]))
    ecart = estimation / infos["loyer_median"] - 1
    k2.metric("Ce bien par rapport au quartier", f"{ecart:+.0%}".replace("+", "+ ").replace("-", "− "))
    k3.metric("Annonces dans ce quartier", f"{annonces_quartier}")

    comparables = df[df["Quartier nettoye"] == quartier].copy()
    memes_chambres = comparables[comparables["Chambres numeriques"] == chambres]
    if len(memes_chambres) >= 3:
        comparables, titre = memes_chambres, f"Annonces comparables · {chambres} chambre{'s' if chambres > 1 else ''} à {quartier_affiche}"
    else:
        titre = f"Annonces à {quartier_affiche}"
    comparables = comparables.assign(distance=(comparables[TARGET] - estimation).abs()).sort_values("distance").head(8)
    if not comparables.empty:
        st.markdown(f"#### {titre}")
        tableau = pd.DataFrame(
            {
                "Annonce": comparables["Titre"].str.strip().str.capitalize(),
                "Chambres": comparables["Chambres numeriques"].map(lambda v: "" if pd.isna(v) else str(int(v))),
                "Standing": comparables["Standing"].map({"Standard": "Standard", "Meuble": "Meublé", "Luxe": "Haut standing"}),
                "Loyer demandé": comparables[TARGET].map(fcfa),
            }
        )
        st.dataframe(tableau, hide_index=True, use_container_width=True)
        st.caption("Loyers demandés dans les annonces, avant négociation.")


# --------------------------------------------------------------------------- onglet 2 : marche
def couleur_loyer(loyer: float) -> str:
    return RAMPE[int(np.searchsorted(PALIERS, loyer, side="right"))]


def carte_des_loyers(quartiers: pd.DataFrame) -> str:
    points = quartiers[(quartiers["annonces"] >= MIN_ANNONCES_CARTE) & quartiers["Latitude"].notna()]
    carte = folium.Map(location=[14.725, -17.46], zoom_start=12, tiles="OpenStreetMap", control_scale=True)
    for _, q in points.sort_values("annonces", ascending=False).iterrows():
        folium.CircleMarker(
            location=[q["Latitude"], q["Longitude"]],
            radius=5 + 1.5 * np.sqrt(q["annonces"]),
            color="#FFFFFF",
            weight=2,
            fill=True,
            fill_color=couleur_loyer(q["loyer_median"]),
            fill_opacity=0.9,
            tooltip=folium.Tooltip(
                f"<b>{q['Quartier']}</b><br>Loyer médian : {fcfa(q['loyer_median'])}<br>{int(q['annonces'])} annonces"
            ),
        ).add_to(carte)
    return carte.get_root().render()


def onglet_marche(df: pd.DataFrame) -> None:
    quartiers = resume_quartiers()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annonces analysées", f"{len(df):,}".replace(",", " "))
    k2.metric("Loyer médian", fcfa(df[TARGET].median()))
    k3.metric("Quartiers couverts", f"{df['Quartier nettoye'].nunique()}")
    k4.metric("Quartier le plus cher", quartiers[quartiers["annonces"] >= MIN_ANNONCES_GRAPHIQUE].sort_values("loyer_median").iloc[-1]["Quartier"])

    st.markdown("#### Carte des loyers médians par quartier")
    components.html(carte_des_loyers(quartiers), height=500)
    bornes = ["Moins de 300 000", "300 000 à 500 000", "500 000 à 800 000", "800 000 à 1,2 million", "Plus de 1,2 million"]
    legende = "".join(f'<span><i style="background:{c}"></i>{b}</span>' for c, b in zip(RAMPE, bornes))
    st.markdown(
        f'<div class="legende"><span>Loyer médian (F CFA) :</span>{legende}<span>· Taille du cercle : nombre d\'annonces</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Les quartiers du plus cher au plus abordable")
    classement = quartiers[quartiers["annonces"] >= MIN_ANNONCES_GRAPHIQUE].copy()
    classement["Loyer médian"] = classement["loyer_median"].map(fcfa)
    graphique = (
        alt.Chart(classement)
        .mark_bar(color=COULEUR, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=16)
        .encode(
            x=alt.X(
                "loyer_median:Q",
                title="Loyer médian mensuel (F CFA)",
                axis=alt.Axis(labelExpr="datum.value == 0 ? '0' : datum.value >= 1000000 ? replace(format(datum.value / 1000000, '.1f'), '.', ',') + ' million' + (datum.value >= 2000000 ? 's' : '') : format(datum.value / 1000, '.0f') + ' 000'", grid=True, tickCount=6),
            ),
            y=alt.Y("Quartier:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            tooltip=[
                alt.Tooltip("Quartier:N"),
                alt.Tooltip("Loyer médian:N"),
                alt.Tooltip("annonces:Q", title="Annonces"),
            ],
        )
        .properties(height=26 * len(classement))
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=13, titleFontSize=12, titleColor="#52606D", labelColor="#1F2933", gridColor="#E4E9E8", domain=False, tickSize=0, labelPadding=8)
    )
    st.altair_chart(graphique, use_container_width=True)
    st.caption(f"Quartiers comptant au moins {MIN_ANNONCES_GRAPHIQUE} annonces. Le loyer médian partage les annonces en deux moitiés égales.")

    st.markdown("#### Loyers selon la taille du logement")
    tailles = df.dropna(subset=["Chambres numeriques"])
    tailles = tailles[tailles["Chambres numeriques"].between(1, 5)]
    par_taille = tailles.groupby("Chambres numeriques")[TARGET].agg(
        annonces="size", median="median", bas=lambda s: s.quantile(0.25), haut=lambda s: s.quantile(0.75)
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Taille": [f"{int(n)} chambre{'s' if n > 1 else ''}" for n in par_taille.index],
                "Annonces": par_taille["annonces"].to_numpy(),
                "Loyer médian": par_taille["median"].map(fcfa).to_numpy(),
                "La moitié des loyers se situe entre": [f"{fcfa(arrondi(b))} et {fcfa(arrondi(h))}" for b, h in zip(par_taille["bas"], par_taille["haut"])],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------------------------------- onglet 3 : a propos
def onglet_a_propos(df: pd.DataFrame, fiabilite: dict) -> None:
    collecte = pd.to_datetime(df["Date collecte"], errors="coerce").max() if "Date collecte" in df.columns else pd.NaT
    mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
    date_collecte = f"{mois[collecte.month - 1]} {collecte.year}" if pd.notna(collecte) else "2026"

    gauche, droite = st.columns(2)
    with gauche:
        st.markdown("#### D'où viennent les chiffres")
        st.markdown(
            f"""
L'outil s'appuie sur **{len(df):,} annonces d'appartements à louer** publiées sur Expat-Dakar et relevées en {date_collecte},
dans {df['Quartier nettoye'].nunique()} quartiers de Dakar et de sa périphérie.

Avant analyse, les annonces qui ne sont pas des loyers mensuels de logements ont été écartées :
locations à la journée, bureaux et commerces, ventes, doublons et annonces sans prix.
""".replace(",", " ", 1)
        )
        st.markdown("#### Comment l'estimation est calculée")
        st.markdown(
            """
L'estimateur a appris, à partir de ces annonces, comment le loyer varie selon **le quartier, la taille,
le type de bien et le standing**. Pour un bien donné, il propose le loyer le plus cohérent avec le marché observé.

Le quartier est de loin le facteur qui pèse le plus, suivi du nombre de chambres.
"""
        )
    with droite:
        st.markdown("#### Quelle confiance accorder à l'estimation")
        st.markdown(
            f"""
La fiabilité a été mesurée en estimant des annonces que l'outil n'avait jamais vues, puis en comparant avec le loyer réellement demandé :

- **{round(fiabilite['part_20'] * 10)} estimations sur 10** tombent à moins de 20 % du loyer demandé ;
- **{round(fiabilite['part_30'] * 10)} sur 10** à moins de 30 % ;
- l'écart typique est d'environ **{fiabilite['ecart_median']:.0%}**.

C'est pourquoi l'outil affiche toujours une fourchette, et pas seulement un chiffre.
"""
        )
        st.markdown("#### Ce que l'outil ne voit pas")
        st.markdown(
            """
- **La surface, l'étage, l'état du bien et la vue** ne figurent pas dans les annonces relevées : deux biens du même profil peuvent donc avoir des loyers différents.
- Ce sont des **loyers demandés**, avant négociation.
- Les biens haut de gamme sont plus nombreux en ligne que sur le marché réel ; les quartiers populaires sont moins bien couverts.
- Pour les biens de très haut standing (plus de 2 millions par mois), l'outil a tendance à **sous-estimer**.
"""
        )


# --------------------------------------------------------------------------- page
def main() -> None:
    st.set_page_config(page_title="Loyers à Dakar", page_icon="🏠", layout="wide")
    habillage()
    df = charger_annonces()
    modele, fiabilite = entrainer()

    estimation, marche, a_propos = st.tabs(["Estimer un loyer", "Le marché par quartier", "À propos"])
    with estimation:
        onglet_estimation(df, modele, fiabilite)
    with marche:
        onglet_marche(df)
    with a_propos:
        onglet_a_propos(df, fiabilite)


if __name__ == "__main__":
    main()
