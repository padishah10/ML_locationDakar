from __future__ import annotations
import csv
import re
import unicodedata
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

from project_paths import RAW_LISTINGS_CSV, ensure_data_dir

REQUEST_TIMEOUT = 15
MAX_PAGES = 150
MAX_EMPTY_PAGES = 3

CATEGORY_CONFIGS = [
    {
        "categorie_annonce": "Appartement",
        "url_template": "https://www.expat-dakar.com/appartements-a-louer/dakar?page={page}",
    },
    {
        "categorie_annonce": "Chambre",
        "url_template": "https://www.expat-dakar.com/chambres-a-louer/dakar?page={page}",
    },
]

DAYS_OF_WEEK = {
    "lundi": 0,
    "mardi": 1,
    "mercredi": 2,
    "jeudi": 3,
    "vendredi": 4,
    "samedi": 5,
    "dimanche": 6,
}

MONTHS = {
    "jan": 1,
    "janv": 1,
    "janvier": 1,
    "fev": 2,
    "fevr": 2,
    "fevrier": 2,
    "fvr": 2,
    "mar": 3,
    "mars": 3,
    "avr": 4,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7,
    "juillet": 7,
    "aou": 8,
    "aout": 8,
    "sept": 9,
    "sep": 9,
    "septembre": 9,
    "oct": 10,
    "octobre": 10,
    "nov": 11,
    "novembre": 11,
    "dec": 12,
    "decembre": 12,
}


def normalize_text(value: str | None) -> str:
    if not value:
        return "N/A"
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_month_token(value: str) -> str:
    token = strip_accents(value.lower())
    return re.sub(r"[^a-z]", "", token)


def convert_weekday_to_date(day_text: str, scrape_date: date) -> date | None:
    weekday = DAYS_OF_WEEK.get(day_text)
    if weekday is None:
        return None
    delta = (scrape_date.weekday() - weekday) % 7 or 7
    return scrape_date - timedelta(days=delta)


def convert_absolute_date(text: str, scrape_date: date) -> date | None:
    match = re.search(r"(\d{1,2})\.?\s+([\w.]+)", text)
    if not match:
        return None

    day = int(match.group(1))
    month_token = normalize_month_token(match.group(2))
    month = MONTHS.get(month_token)
    if month is None:
        return None

    year = scrape_date.year
    try:
        parsed_date = date(year, month, day)
    except ValueError:
        return None

    if parsed_date > scrape_date:
        try:
            parsed_date = date(year - 1, month, day)
        except ValueError:
            return None
    return parsed_date


def convert_publication_date(raw_text: str, scrape_date: date) -> str:
    text = normalize_text(raw_text).lower()
    text = text.replace("’", "'")
    text = re.sub(r"\d{1,2}:\d{2}", "", text)
    text = text.replace(",", " ").strip()

    if "aujourd'hui" in text or "aujourdhui" in text:
        return scrape_date.isoformat()
    if "hier" in text:
        return (scrape_date - timedelta(days=1)).isoformat()

    first_token = text.split()[0] if text else ""
    weekday_date = convert_weekday_to_date(first_token, scrape_date)
    if weekday_date is not None:
        return weekday_date.isoformat()

    absolute_date = convert_absolute_date(text, scrape_date)
    if absolute_date is not None:
        return absolute_date.isoformat()

    return "Inconnu"


def extract_listing(annonce, scrape_date: date, categorie_annonce: str) -> list[str]:
    title_tag = annonce.find("div", class_="listing-card__header__title")
    location_tag = annonce.find("div", class_="listing-card__header__location")
    price_tag = annonce.select_one("span.listing-card__price__value")
    bedrooms_tag = annonce.select_one("span.listing-card__header__tags__item--no-of-bedrooms")
    date_tag = annonce.select_one("div.listing-card__header__date")

    title = normalize_text(title_tag.get_text()) if title_tag else "N/A"
    location = normalize_text(location_tag.get_text()) if location_tag else "N/A"
    location = re.sub(r",\s*Dakar$", "", location, flags=re.IGNORECASE)
    price = normalize_text(price_tag.get_text()) if price_tag else "N/A"
    bedrooms = normalize_text(bedrooms_tag.get_text()) if bedrooms_tag else "N/A"
    raw_date = normalize_text(date_tag.get_text()) if date_tag else "N/A"
    parsed_date = convert_publication_date(raw_date, scrape_date)

    # Lien de l'annonce : identifiant stable pour reperer une annonce deja collectee.
    link = normalize_text(annonce.get("href")) if annonce.get("href") else "N/A"

    return [categorie_annonce, title, location, price, bedrooms, raw_date, parsed_date, scrape_date.isoformat(), link]


def scrape_category(session: requests.Session, categorie_annonce: str, url_template: str, max_pages: int) -> list[list[str]]:
    scrape_date = datetime.utcnow().date()
    rows: list[list[str]] = []
    empty_pages = 0

    for page in range(1, max_pages + 1):
        url = url_template.format(page=page)
        print(f"Scraping {categorie_annonce.lower()} page {page}: {url}")

        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            print(f"Request failed for {categorie_annonce.lower()} page {page}: {exc}")
            if status_code == 404:
                break
            continue
        except requests.RequestException as exc:
            print(f"Request failed for {categorie_annonce.lower()} page {page}: {exc}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        annonces = soup.find_all("a", class_="listing-card__inner")
        if not annonces:
            empty_pages += 1
            if empty_pages >= MAX_EMPTY_PAGES:
                break
            continue

        empty_pages = 0
        rows.extend(extract_listing(annonce, scrape_date, categorie_annonce) for annonce in annonces)

    return rows


def scrape_listings(max_pages: int = MAX_PAGES) -> list[list[str]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
        }
    )

    rows: list[list[str]] = []
    for category_config in CATEGORY_CONFIGS:
        rows.extend(
            scrape_category(
                session=session,
                categorie_annonce=category_config["categorie_annonce"],
                url_template=category_config["url_template"],
                max_pages=max_pages,
            )
        )
    return rows


COLUMNS = [
    "Categorie annonce",
    "Titre",
    "Lieu",
    "Prix",
    "Chambres",
    "Date brute",
    "Date estimee",
    "Date collecte",
    "Lien",
]


def load_previous_rows() -> list[list[str]]:
    """Relit les collectes precedentes pour les cumuler avec la nouvelle.

    La premiere version ecrasait le fichier a chaque execution : une seule
    collecte etait conservee. Les fichiers de l'ancien format (sans date de
    collecte ni lien) sont acceptes : les colonnes manquantes sont laissees vides.
    """
    if not RAW_LISTINGS_CSV.exists():
        return []
    with RAW_LISTINGS_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return [[(record.get(column) or "") for column in COLUMNS] for record in reader]


def merge_rows(previous: list[list[str]], new: list[list[str]]) -> list[list[str]]:
    """Cumule les collectes. Une annonce deja presente (meme lien, ou a defaut
    meme categorie, titre, lieu, prix et date de publication) n'est pas ajoutee
    une seconde fois : on garde sa premiere observation."""
    lien_index = COLUMNS.index("Lien")

    def key(row: list[str]):
        lien = row[lien_index]
        if lien and lien != "N/A":
            return ("lien", lien)
        return ("contenu", row[0], row[1], row[2], row[3], row[6])

    seen = set()
    merged: list[list[str]] = []
    for row in previous + new:
        row_key = key(row)
        if row_key in seen:
            continue
        seen.add(row_key)
        merged.append(row)
    return merged


def save_rows(rows: list[list[str]]) -> None:
    ensure_data_dir()
    with RAW_LISTINGS_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def main() -> None:
    new_rows = scrape_listings()
    if not new_rows:
        print("Aucune annonce collectee : le fichier existant est conserve tel quel.")
        return
    previous_rows = load_previous_rows()
    rows = merge_rows(previous_rows, new_rows)
    save_rows(rows)
    print(
        f"{len(new_rows)} annonces collectees, {len(rows) - len(previous_rows)} nouvelles ; "
        f"{len(rows)} annonces au total dans {RAW_LISTINGS_CSV.name}"
    )


if __name__ == "__main__":
    main()

