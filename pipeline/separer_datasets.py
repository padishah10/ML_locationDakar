"""Filtrage des annonces et separation appartements / chambres.

Les donnees viennent d'un scraping : la plateforme laisse passer des annonces
qui ne sont pas des loyers mensuels de logements. Chaque annonce ecartee recoit
un motif d'exclusion, enregistre dans annonces_exclues.csv, pour que le
nettoyage soit tracable et reproductible a chaque nouvelle collecte.
"""
from __future__ import annotations
import re
import unicodedata

import pandas as pd

from project_paths import (
    APPARTEMENTS_ML_CSV,
    CHAMBRES_ML_CSV,
    EXCLUDED_LISTINGS_CSV,
    GEO_LISTINGS_CSV,
    ensure_data_dir,
)

# Seuils (F CFA). Ils sont fixes, et non plus calcules sur un quantile :
# un quantile change a chaque collecte et supprimait a tort des biens de luxe reels.
PRIX_SYMBOLIQUE_MAX = 5000          # "1 F CFA" = prix non communique
PRIX_VENTE_MIN = 20_000_000         # au-dela : prix de vente, pas un loyer
LOYER_MENSUEL_MIN = {"appartement": 60_000, "chambre": 45_000}  # en dessous : tarif a la journee

MOTIF_COLUMN = "Motif exclusion"
CLES_DOUBLON = ["Categorie annonce", "Titre", "Lieu", "Prix numerique", "Chambres numeriques"]

VENTE_PATTERN = re.compile(r"\ba\s+vend(re|e|u)\b|\bvente\b")
PRO_PATTERN = re.compile(r"\b(bureaux?|commerces?|magasins?|entrepots?|local commercial|locaux)\b")
HABITATION_PATTERN = re.compile(r"\bhabitation\b|\bchambres?\b")
JOURNALIER_PATTERN = re.compile(r"\b(par|a la|la)\s+(jours?|journee|nuits?|nuitees?)\b|\bnuitees?\b|\bhotel\b")
MENSUEL_PATTERN = re.compile(r"\bpar\s+mois\b|\bmensuel")
LOGEMENT_ENTIER_PATTERN = re.compile(r"\bappartements?\b|\b[ft]\s?[2-9]\b")


def cle_texte(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def motif_exclusion(row: pd.Series) -> str:
    """Retourne le motif d'exclusion d'une annonce, ou une chaine vide si elle est conservee."""
    categorie = cle_texte(row["Categorie annonce"])
    titre = cle_texte(row["Titre"])
    prix = row["Prix numerique"]

    if pd.isna(prix):
        return "Prix absent"
    if not cle_texte(row.get("Quartier nettoye")) or row.get("Quartier nettoye") == "N/A":
        return "Quartier absent"
    if prix <= PRIX_SYMBOLIQUE_MAX:
        return "Prix symbolique (non communique)"
    if prix >= PRIX_VENTE_MIN or (VENTE_PATTERN.search(titre) and "louer" not in titre):
        return "Annonce de vente"
    # "3 chambres + bureau" ou "usage habitation ou bureau" restent des logements.
    if PRO_PATTERN.search(titre) and not HABITATION_PATTERN.search(titre):
        return "Local professionnel"
    if categorie == "chambre" and LOGEMENT_ENTIER_PATTERN.search(titre):
        return "Categorie incoherente (logement entier dans Chambres)"
    seuil = LOYER_MENSUEL_MIN.get(categorie)
    if JOURNALIER_PATTERN.search(titre) and not MENSUEL_PATTERN.search(titre):
        return "Location a la journee"
    if seuil is not None and prix < seuil and not MENSUEL_PATTERN.search(titre):
        return "Location a la journee"
    return ""


def marquer_doublons(df: pd.DataFrame) -> pd.Series:
    """Une meme annonce republiee (meme titre, lieu, prix, chambres) n'est gardee qu'une fois : la plus recente."""
    ordre = df.sort_values("Date NLP", ascending=False, kind="stable")
    cles = ordre[CLES_DOUBLON].copy()
    cles["Titre"] = cles["Titre"].map(cle_texte)
    cles["Lieu"] = cles["Lieu"].map(cle_texte)
    return cles.duplicated(keep="first").reindex(df.index)


def filtrer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[MOTIF_COLUMN] = df.apply(motif_exclusion, axis=1)
    valides = df[MOTIF_COLUMN] == ""
    doublons = marquer_doublons(df[valides])
    df.loc[doublons[doublons].index, MOTIF_COLUMN] = "Doublon (annonce republiee)"
    return df


def main() -> None:
    ensure_data_dir()
    df = pd.read_csv(GEO_LISTINGS_CSV, sep=";", encoding="utf-8-sig")

    if "Categorie annonce" not in df.columns:
        raise KeyError("La colonne 'Categorie annonce' est absente. Relance d'abord webscraping.py puis le pipeline.")

    df = filtrer(df)
    conserve = df[df[MOTIF_COLUMN] == ""].drop(columns=[MOTIF_COLUMN])
    conserve["Prix numerique"] = conserve["Prix numerique"].astype("Int64")
    categorie = conserve["Categorie annonce"].astype(str).str.lower()
    appartements = conserve[categorie == "appartement"]
    chambres = conserve[categorie == "chambre"]

    appartements.to_csv(APPARTEMENTS_ML_CSV, sep=";", index=False, encoding="utf-8-sig")
    chambres.to_csv(CHAMBRES_ML_CSV, sep=";", index=False, encoding="utf-8-sig")
    df[df[MOTIF_COLUMN] != ""].to_csv(EXCLUDED_LISTINGS_CSV, sep=";", index=False, encoding="utf-8-sig")

    print(f"Annonces brutes : {len(df)}")
    bilan = df[df[MOTIF_COLUMN] != ""].groupby(["Categorie annonce", MOTIF_COLUMN]).size()
    print(bilan.to_string())
    print(f"Dataset appartements: {len(appartements)} lignes -> {APPARTEMENTS_ML_CSV.name}")
    print(f"Dataset chambres: {len(chambres)} lignes -> {CHAMBRES_ML_CSV.name}")


if __name__ == "__main__":
    main()
