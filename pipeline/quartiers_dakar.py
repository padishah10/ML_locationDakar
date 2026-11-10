from __future__ import annotations
import time

import pandas as pd
from geopy.geocoders import Nominatim

from project_paths import GEO_LISTINGS_CSV, NLP_LISTINGS_CSV, QUARTIERS_CSV

REQUEST_DELAY_SECONDS = 1


def load_cache() -> pd.DataFrame:
    if QUARTIERS_CSV.exists():
        return pd.read_csv(QUARTIERS_CSV, sep=";", encoding="utf-8-sig")
    return pd.DataFrame(columns=["Quartier nettoye", "Latitude", "Longitude"])


def get_coords(geolocator: Nominatim, quartier: str) -> tuple[float | None, float | None]:
    try:
        location = geolocator.geocode(f"{quartier}, Dakar, Senegal", timeout=20)
    except Exception:
        return None, None
    if location is None:
        return None, None
    return location.latitude, location.longitude


def update_cache(quartiers: list[str]) -> pd.DataFrame:
    cache = load_cache().drop_duplicates(subset=["Quartier nettoye"]).copy()
    geolocator = Nominatim(user_agent="memoire_dakar_ml")

    known = set(cache["Quartier nettoye"].dropna().astype(str))
    missing = [quartier for quartier in quartiers if quartier and quartier != "N/A" and quartier not in known]

    if not missing:
        return cache

    rows = []
    for quartier in missing:
        latitude, longitude = get_coords(geolocator, quartier)
        rows.append(
            {
                "Quartier nettoye": quartier,
                "Latitude": latitude,
                "Longitude": longitude,
            }
        )
        time.sleep(REQUEST_DELAY_SECONDS)

    updated_cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
    updated_cache = updated_cache.drop_duplicates(subset=["Quartier nettoye"], keep="last")
    updated_cache.to_csv(QUARTIERS_CSV, sep=";", index=False, encoding="utf-8-sig")
    return updated_cache


def main() -> None:
    df = pd.read_csv(NLP_LISTINGS_CSV, sep=";", encoding="utf-8-sig")
    quartiers = sorted(df["Quartier nettoye"].dropna().astype(str).unique())
    cache = update_cache(quartiers)

    merged = df.merge(cache, on="Quartier nettoye", how="left")
    merged.to_csv(GEO_LISTINGS_CSV, sep=";", index=False, encoding="utf-8-sig")
    print(f"Coordonnees ajoutees dans {GEO_LISTINGS_CSV.name}")


if __name__ == "__main__":
    main()

