"""Entrainement des modeles de prediction des loyers.

Le modele final est entraine sur toutes les annonces retenues. Ses performances
sont estimees par validation croisee repetee (voir evaluer_modeles.py pour
l'evaluation complete : modeles de reference, comparaison d'algorithmes,
contribution de chaque groupe de variables).
"""
from __future__ import annotations
import json
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from project_paths import (
    APPARTEMENTS_ML_CSV,
    APPARTEMENTS_MODEL,
    CHAMBRES_ML_CSV,
    CHAMBRES_MODEL,
    METRICS_JSON,
    ensure_metrics_dir,
    ensure_models_dir,
)

RANDOM_STATE = 42
CV_SPLITS = 5
CV_REPEATS = 3
MIN_ROWS_REQUIRED = 50
TARGET = "Prix numerique"

# Les variables de date de publication (annee, mois, jour, anciennete) ont ete
# retirees : elles decrivent le moment de la collecte, pas le logement, et la
# validation croisee montre qu'elles degradent la prediction.
NUMERIC_FEATURES = [
    "Chambres numeriques",
    "Latitude",
    "Longitude",
]

CATEGORICAL_FEATURES = [
    "Standing",
    "Type logement",
    "Quartier nettoye",
]

SEGMENT_CONFIG = {
    "appartements": {
        "dataset_path": APPARTEMENTS_ML_CSV,
        "model_path": APPARTEMENTS_MODEL,
        "label": "Appartements",
    },
    "chambres": {
        "dataset_path": CHAMBRES_ML_CSV,
        "model_path": CHAMBRES_MODEL,
        "label": "Chambres",
    },
}


def load_dataset(path):
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def build_preprocessor(numeric_features=None, categorical_features=None) -> ColumnTransformer:
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    transformers = []
    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), list(numeric_features)))
    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, list(categorical_features)))
    return ColumnTransformer(transformers=transformers)


def build_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_model_pipeline(regressor=None, numeric_features=None, categorical_features=None) -> Pipeline:
    """Pretraitement + regresseur entraine sur le logarithme du loyer."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=build_regressor() if regressor is None else regressor,
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def compute_scores(y_true, y_pred) -> dict:
    """Metriques d'un pli de validation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_log": float(r2_score(np.log(y_true), np.log(np.clip(y_pred, 1, None)))),
        "erreur_relative_mediane_pct": float(np.median(ape) * 100),
        "part_a_20pct": float((ape <= 0.20).mean() * 100),
    }


def cross_validate_model(make_model, x: pd.DataFrame, y: pd.Series) -> dict:
    """Validation croisee repetee : moyenne et ecart-type de chaque metrique."""
    splitter = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    y_values = y.astype(float).to_numpy()
    fold_scores = []
    for train_index, test_index in splitter.split(x):
        model = make_model()
        model.fit(x.iloc[train_index], y_values[train_index])
        fold_scores.append(compute_scores(y_values[test_index], model.predict(x.iloc[test_index])))
    scores = pd.DataFrame(fold_scores)
    result = {name: round(float(value), 4) for name, value in scores.mean().items()}
    result.update({f"{name}_ecart_type": round(float(value), 4) for name, value in scores.std(ddof=1).items()})
    return result


def evaluate_segment(df: pd.DataFrame, model_output_path, segment_name: str) -> dict:
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_columns = set(feature_columns) - set(df.columns)
    if missing_columns:
        raise KeyError(f"Colonnes manquantes pour {segment_name}: {sorted(missing_columns)}")

    x = df[feature_columns]
    y = df[TARGET].astype(float)

    cv_scores = cross_validate_model(build_model_pipeline, x, y)

    # Modele final : entraine sur toutes les annonces, pour l'application.
    pipeline = build_model_pipeline()
    pipeline.fit(x, y)
    ensure_models_dir()
    joblib.dump(pipeline, model_output_path)

    return {
        "segment": segment_name,
        "lignes_total": int(len(df)),
        "validation": f"validation croisee {CV_SPLITS} plis x {CV_REPEATS} repetitions",
        "mae": round(cv_scores["mae"], 2),
        "rmse": round(cv_scores["rmse"], 2),
        "r2": cv_scores["r2"],
        "r2_ecart_type": cv_scores["r2_ecart_type"],
        "r2_log": cv_scores["r2_log"],
        "erreur_relative_mediane_pct": round(cv_scores["erreur_relative_mediane_pct"], 2),
        "part_a_20pct": round(cv_scores["part_a_20pct"], 2),
        "prix_min": int(df[TARGET].min()),
        "prix_median": int(df[TARGET].median()),
        "prix_max": int(df[TARGET].max()),
        "modele": model_output_path.name,
    }


def train_all_models() -> dict:
    metrics = {}
    for segment_name, config in SEGMENT_CONFIG.items():
        df = load_dataset(config["dataset_path"])
        if len(df) < MIN_ROWS_REQUIRED:
            # Trop peu d'annonces pour entrainer et evaluer honnetement un modele.
            print(
                f"Segment {segment_name} ignore : {len(df)} annonces "
                f"(minimum {MIN_ROWS_REQUIRED}). Aucun modele entraine."
            )
            if config["model_path"].exists():
                config["model_path"].unlink()
            continue
        metrics[segment_name] = evaluate_segment(df, config["model_path"], segment_name)

    ensure_metrics_dir()
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train_all_models()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
