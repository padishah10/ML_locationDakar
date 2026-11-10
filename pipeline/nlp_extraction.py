from __future__ import annotations
import re
import unicodedata
from datetime import datetime, timedelta

import dateparser
import pandas as pd

from project_paths import NLP_LISTINGS_CSV, RAW_LISTINGS_CSV

SOURCE_DATE_COLUMN_CANDIDATES = ["Date brute", "Date brute ", "Date brute"]
ESTIMATED_DATE_COLUMN_CANDIDATES = ["Date estimee", "Date estimée"]
CATEGORY_COLUMN_CANDIDATES = ["Categorie annonce", "Catégorie annonce"]


def normalize_text(value: str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_token(value: str) -> str:
    text = strip_accents(value.lower())
    return re.sub(r"[^a-z0-9 ]", " ", text)


def pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Missing expected column among: {candidates}")


def corriger_mois_corrompus(text: str) -> str:
    replacements = {
        "aout": "aout",
        "août": "aout",
        "fevr": "fevrier",
        "fevrier": "fevrier",
        "fvr": "fevrier",
        "déc": "decembre",
        "dec.": "decembre",
        "oct.": "octobre",
        "sept.": "septembre",
        "nov.": "novembre",
        "janv.": "janvier",
        "fevr.": "fevrier",
        "avr.": "avril",
        "juil.": "juillet",
    }
    normalized = normalize_token(text)
    for wrong, right in replacements.items():
        normalized = normalized.replace(normalize_token(wrong), right)
    return normalized


def convertir_date_nlp(raw_text: str, fallback_text: str | None = None, base_date: datetime | None = None) -> str:
    """Convertit la date brute en date ISO.

    Les dates relatives ("hier", "lundi") n'ont de sens que par rapport au jour
    du scraping. La date calculee par le scraper ("Date estimee") est donc
    prioritaire ; on ne re-interprete le texte brut que si elle est absente,
    et toujours par rapport a la date de collecte, jamais a la date du jour.
    """
    if fallback_text:
        fallback = normalize_text(fallback_text)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", fallback):
            return fallback
    if base_date is None:
        base_date = datetime.utcnow()
    text = normalize_text(raw_text).lower()
    text = corriger_mois_corrompus(text)
    text = re.sub(r"\d{1,2}:\d{2}", "", text).strip(" ,")

    if not text and fallback_text:
        fallback = normalize_text(fallback_text)
        if fallback and fallback.lower() != "inconnu":
            return fallback
    if not text:
        return "Inconnu"

    if "aujourd'hui" in text or "aujourdhui" in text:
        return base_date.date().isoformat()
    if "hier" in text:
        return (base_date - timedelta(days=1)).date().isoformat()

    weekdays = {
        "lundi": 0,
        "mardi": 1,
        "mercredi": 2,
        "jeudi": 3,
        "vendredi": 4,
        "samedi": 5,
        "dimanche": 6,
    }
    for day, index in weekdays.items():
        if day in text:
            delta = (base_date.weekday() - index) % 7 or 7
            return (base_date - timedelta(days=delta)).date().isoformat()

    parsed = dateparser.parse(
        text,
        languages=["fr"],
        settings={
            "RELATIVE_BASE": base_date,
            "PREFER_DAY_OF_MONTH": "first",
            "PREFER_DATES_FROM": "past",
            "DATE_ORDER": "DMY",
        },
    )
    if parsed is not None:
        return parsed.date().isoformat()

    if fallback_text:
        fallback = normalize_text(fallback_text)
        if fallback and fallback.lower() != "inconnu":
            return fallback
    return "Inconnu"


TYPE_PIECES_PATTERN = re.compile(r"\b([ft])\s?([1-9])\b")


def extraire_type(title: str, categorie_annonce: str = "") -> str:
    """Type de logement deduit du titre.

    Corrections par rapport a la premiere version :
    - le mot "chambre" dans un titre d'appartement ("Appartement 3 chambres")
      ne transforme plus le bien en "Chambre" : seule la categorie de la
      plateforme decide qu'une annonce est une chambre ;
    - T3 et F3 designent le meme bien : tout est ramene a la notation F.
    """
    normalized = normalize_token(normalize_text(title))
    categorie = normalize_token(normalize_text(categorie_annonce))
    if "chambre" in categorie:
        return "Chambre"
    if not normalized:
        return "Appartement"
    if any(term in normalized for term in ["penthouse", "appartement terrasse", "appartementterrasse", "penthouss", "penthous"]):
        return "Penthouse"
    room_match = TYPE_PIECES_PATTERN.search(normalized)
    if room_match:
        return f"F{room_match.group(2)}"
    if "studio" in normalized:
        return "Studio"
    if "villa" in normalized:
        return "Villa"
    if "duplex" in normalized:
        return "Duplex"
    if "immeuble" in normalized:
        return "Immeuble"
    return "Appartement"


NON_MEUBLE_PATTERN = re.compile(r"\b(non|pas|sans)\s+(meubl\w*|equip\w*)")
MEUBLE_PATTERN = re.compile(r"\bmeubl\w*")
LUXE_TERMS = [
    "penthouse",
    "penthouss",
    "penthous",
    "appartement terrasse",
    "haut standing",
    "grand standing",
    "luxe",
    "luxueux",
    "luxueuse",
]


def est_meuble(title: str) -> bool:
    """Vrai si le titre annonce un bien meuble.

    "non meuble", "sans meubles", "pas meuble" sont des negations : la premiere
    version les classait a tort en "Meuble" (11 annonces sur 12).
    "equipe" seul n'est plus retenu : "cuisine equipee" ne dit rien du mobilier.
    """
    normalized = normalize_token(normalize_text(title))
    normalized = re.sub(r"\s+", " ", normalized)
    sans_negation = NON_MEUBLE_PATTERN.sub(" ", normalized)
    return bool(MEUBLE_PATTERN.search(sans_negation))


def extraire_standing(title: str) -> str:
    normalized = normalize_token(normalize_text(title))
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return "Standard"
    if any(term in normalized for term in LUXE_TERMS):
        return "Luxe"
    if est_meuble(title):
        return "Meuble"
    return "Standard"


def extraire_prix_numerique(price: str):
    text = normalize_text(price)
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else pd.NA


NOMBRES_EN_LETTRES = {"une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6}


def extraire_chambres_numeriques(value: str, title: str, categorie_annonce: str = ""):
    categorie = normalize_token(normalize_text(categorie_annonce))
    text = normalize_text(value)
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))

    normalized_title = normalize_token(normalize_text(title))
    # Usage dakarois : un F4 = 3 chambres + 1 salon.
    room_match = TYPE_PIECES_PATTERN.search(normalized_title)
    if room_match:
        return max(int(room_match.group(2)) - 1, 1)
    # "Appartement 3 chambres", "deux chambres salon"...
    count_match = re.search(r"\b(\d|une|deux|trois|quatre|cinq|six)\s+chambres?\b", normalized_title)
    if count_match:
        token = count_match.group(1)
        return int(token) if token.isdigit() else NOMBRES_EN_LETTRES[token]
    if "studio" in normalized_title:
        return 1
    if "chambre" in categorie:
        return 1
    return pd.NA


def enrich_dataset(df: pd.DataFrame) -> pd.DataFrame:
    raw_date_column = pick_existing_column(df, SOURCE_DATE_COLUMN_CANDIDATES)
    estimated_date_column = pick_existing_column(df, ESTIMATED_DATE_COLUMN_CANDIDATES)
    category_column = pick_existing_column(df, CATEGORY_COLUMN_CANDIDATES)

    enriched = df.copy()
    enriched[category_column] = enriched[category_column].fillna("").map(normalize_text)
    enriched["Titre"] = enriched["Titre"].fillna("").map(normalize_text)
    enriched["Lieu"] = enriched["Lieu"].fillna("").map(normalize_text)
    enriched["Prix"] = enriched["Prix"].fillna("").map(normalize_text)
    enriched["Chambres"] = enriched["Chambres"].fillna("").map(normalize_text)
    enriched[raw_date_column] = enriched[raw_date_column].fillna("").map(normalize_text)

    enriched["Type logement"] = [
        extraire_type(titre, categorie_annonce)
        for titre, categorie_annonce in zip(enriched["Titre"], enriched[category_column])
    ]
    enriched["Standing"] = enriched["Titre"].apply(extraire_standing)
    enriched["Prix numerique"] = enriched["Prix"].apply(extraire_prix_numerique).astype("Int64")
    enriched["Chambres numeriques"] = [
        extraire_chambres_numeriques(chambres, titre, categorie_annonce)
        for chambres, titre, categorie_annonce in zip(enriched["Chambres"], enriched["Titre"], enriched[category_column])
    ]
    enriched["Chambres numeriques"] = enriched["Chambres numeriques"].astype("Int64")
    enriched["Meuble"] = enriched["Titre"].apply(lambda titre: "Oui" if est_meuble(titre) else "Non")

    # Date de collecte : colonne ecrite par le scraper (nouvelles collectes) ou,
    # a defaut, date de publication la plus recente du fichier.
    estimated = pd.to_datetime(enriched[estimated_date_column], errors="coerce")
    if "Date collecte" in enriched.columns:
        collecte = pd.to_datetime(enriched["Date collecte"], errors="coerce")
    else:
        collecte = pd.Series(pd.NaT, index=enriched.index)
    collecte = collecte.fillna(estimated[collecte.isna()].max())
    enriched["Date collecte"] = collecte.dt.date.astype(str)

    enriched["Date NLP"] = [
        convertir_date_nlp(raw_date, fallback_date, base_date.to_pydatetime() if pd.notna(base_date) else None)
        for raw_date, fallback_date, base_date in zip(
            enriched[raw_date_column], enriched[estimated_date_column].fillna("").astype(str), collecte
        )
    ]

    parsed_dates = pd.to_datetime(enriched["Date NLP"], errors="coerce")
    # Anciennete de l'annonce au moment de la collecte (et non au moment ou le
    # script est relance) : le resultat ne change plus d'une execution a l'autre.
    enriched["Jours depuis publication"] = (collecte - parsed_dates).dt.days.clip(lower=0).astype("Int64")
    enriched["Annee publication"] = parsed_dates.dt.year.astype("Int64")
    enriched["Mois publication"] = parsed_dates.dt.month.astype("Int64")
    enriched["Jour publication"] = parsed_dates.dt.day.astype("Int64")

    return enriched


def main() -> None:
    df = pd.read_csv(RAW_LISTINGS_CSV, sep=";", encoding="utf-8-sig")
    enriched = enrich_dataset(df)
    enriched.to_csv(NLP_LISTINGS_CSV, sep=";", index=False, encoding="utf-8-sig")
    print(f"Dataset enrichi enregistre dans {NLP_LISTINGS_CSV.name}")


if __name__ == "__main__":
    main()
