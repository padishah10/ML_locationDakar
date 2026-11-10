"""Relance toute la chaine de traitement a partir de annonces_dakar.csv.

Le scraping (webscraping.py) n'est pas inclus : on le lance a part quand on
veut une nouvelle collecte, puis on relance ce script.
"""
import evaluer_modeles
import nlp_extraction
import quartierNettoye
import quartiers_dakar
import separer_datasets
import train_modeles_loyer

ETAPES = [
    ("Extraction des attributs", nlp_extraction.main),
    ("Normalisation des quartiers", quartierNettoye.main),
    ("Geocodage", quartiers_dakar.main),
    ("Filtrage et separation", separer_datasets.main),
    ("Entrainement", train_modeles_loyer.main),
    ("Evaluation", evaluer_modeles.main),
]

if __name__ == "__main__":
    for nom, etape in ETAPES:
        print(f"\n=== {nom} ===")
        etape()
