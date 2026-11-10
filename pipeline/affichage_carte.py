import folium
import pandas as pd

from project_paths import GEO_LISTINGS_CSV, MAP_HTML


def main() -> None:
    df = pd.read_csv(GEO_LISTINGS_CSV, sep=";", encoding="utf-8-sig")
    quartier_column = "Quartier nettoye" if "Quartier nettoye" in df.columns else "Quartier nettoyé"
    points = df.dropna(subset=["Latitude", "Longitude", quartier_column])[
        [quartier_column, "Latitude", "Longitude"]
    ].drop_duplicates()

    carte = folium.Map(location=[14.6928, -17.4467], zoom_start=12)
    for _, row in points.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=row[quartier_column],
            tooltip=row[quartier_column],
        ).add_to(carte)

    carte.save(MAP_HTML)
    print(f"Carte enregistree dans {MAP_HTML.name}")


if __name__ == "__main__":
    main()

