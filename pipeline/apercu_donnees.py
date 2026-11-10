from __future__ import annotations
import pandas as pd

from project_paths import (
    APPARTEMENTS_ML_CSV,
    CHAMBRES_ML_CSV,
    GEO_LISTINGS_CSV,
    NLP_LISTINGS_CSV,
    RAW_LISTINGS_CSV,
)


def load_best_available_dataset() -> tuple[pd.DataFrame, str]:
    for path in [APPARTEMENTS_ML_CSV, CHAMBRES_ML_CSV, GEO_LISTINGS_CSV, NLP_LISTINGS_CSV, RAW_LISTINGS_CSV]:
        if path.exists():
            return pd.read_csv(path, sep=";", encoding="utf-8-sig"), path.name
    raise FileNotFoundError("Aucun dataset disponible dans donnees/.")


def main() -> None:
    df, source_name = load_best_available_dataset()
    print(f"Source: {source_name}")
    print(f"Lignes: {len(df)} | Colonnes: {len(df.columns)}")
    print()
    print("Colonnes:")
    print(df.columns.tolist())
    print()
    print("Apercu:")
    print(df.head())
    print()

    if "Categorie annonce" in df.columns:
        print("Repartition des categories:")
        print(df["Categorie annonce"].value_counts(dropna=False))
        print()

    if "Prix numerique" in df.columns:
        print("Statistiques du prix:")
        print(df["Prix numerique"].describe())
        print()

    print("Valeurs manquantes (%):")
    missing_ratio = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    print(missing_ratio)


if __name__ == "__main__":
    main()
