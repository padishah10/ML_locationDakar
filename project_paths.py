from pathlib import Path


def _find_project_root() -> Path:
    candidates = [
        Path(__file__).resolve().parent,
        Path.cwd(),
        Path("/mount/src/ml_locationdakar"),
    ]

    for candidate in candidates:
        if (candidate / "Fichiers CSV" / "appartements_dakar_ml.csv").exists():
            return candidate

    return Path(__file__).resolve().parent


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "Fichiers CSV"
MODELS_DIR = PROJECT_ROOT / "Modeles"
METRICS_DIR = PROJECT_ROOT / "Metriques"

RAW_LISTINGS_CSV = DATA_DIR / "annonces_dakar.csv"
NLP_LISTINGS_CSV = DATA_DIR / "annonces_dakar_nlp.csv"
GEO_LISTINGS_CSV = DATA_DIR / "annonces_dakar_geo.csv"
QUARTIERS_CSV = DATA_DIR / "quartiers_dakar.csv"
MAP_HTML = DATA_DIR / "carte_quartiers_dakar.html"

APPARTEMENTS_ML_CSV = DATA_DIR / "appartements_dakar_ml.csv"
CHAMBRES_ML_CSV = DATA_DIR / "chambres_dakar_ml.csv"

APPARTEMENTS_MODEL = MODELS_DIR / "modele_appartements.joblib"
CHAMBRES_MODEL = MODELS_DIR / "modele_chambres.joblib"
METRICS_JSON = METRICS_DIR / "metriques_modeles_loyer.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_metrics_dir() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
