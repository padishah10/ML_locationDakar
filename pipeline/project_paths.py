"""Chemins du projet. Les scripts sont dans pipeline/, les donnees a la racine du depot."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "donnees"
MODELS_DIR = PROJECT_ROOT / "modeles"
METRICS_DIR = PROJECT_ROOT / "resultats"

RAW_LISTINGS_CSV = DATA_DIR / "annonces_dakar.csv"
NLP_LISTINGS_CSV = DATA_DIR / "annonces_dakar_nlp.csv"
GEO_LISTINGS_CSV = DATA_DIR / "annonces_dakar_geo.csv"
QUARTIERS_CSV = DATA_DIR / "quartiers_dakar.csv"
MAP_HTML = DATA_DIR / "carte_quartiers_dakar.html"

APPARTEMENTS_ML_CSV = DATA_DIR / "appartements_dakar_ml.csv"
CHAMBRES_ML_CSV = DATA_DIR / "chambres_dakar_ml.csv"
EXCLUDED_LISTINGS_CSV = DATA_DIR / "annonces_exclues.csv"
EVALUATION_JSON = METRICS_DIR / "evaluation_modeles.json"
EVALUATION_MD = METRICS_DIR / "evaluation_modeles.md"

APPARTEMENTS_MODEL = MODELS_DIR / "modele_appartements.joblib"
CHAMBRES_MODEL = MODELS_DIR / "modele_chambres.joblib"
METRICS_JSON = METRICS_DIR / "metriques_modeles_loyer.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_metrics_dir() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
