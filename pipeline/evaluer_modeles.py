"""Evaluation complete du modele de prediction des loyers (appartements).

Produit Metriques/evaluation_modeles.json et evaluation_modeles.md :
  1. comparaison avec des modeles de reference et d'autres algorithmes ;
  2. contribution de chaque groupe de variables (ablation) ;
  3. importance des variables par permutation ;
  4. erreur selon le niveau de loyer et selon le quartier.
Toutes les mesures viennent de la meme validation croisee repetee (5 plis x 3).
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold

from project_paths import APPARTEMENTS_ML_CSV, EVALUATION_JSON, EVALUATION_MD, ensure_metrics_dir
from train_modeles_loyer import (
    CATEGORICAL_FEATURES,
    CV_REPEATS,
    CV_SPLITS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    build_model_pipeline,
    compute_scores,
    cross_validate_model,
    load_dataset,
)

DATE_FEATURES = ["Jours depuis publication", "Annee publication", "Mois publication", "Jour publication"]
QUARTIER = "Quartier nettoye"


class MedianeGlobale(BaseEstimator, RegressorMixin):
    """Reference 1 : predit toujours le loyer median."""

    def fit(self, x, y):
        self.mediane_ = float(np.median(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mediane_)


class MedianeParQuartier(BaseEstimator, RegressorMixin):
    """Reference 2 : predit le loyer median du quartier (median global si quartier inconnu)."""

    def fit(self, x, y):
        y = pd.Series(np.asarray(y, dtype=float), index=x.index)
        self.mediane_ = float(y.median())
        self.par_quartier_ = y.groupby(x[QUARTIER]).median()
        return self

    def predict(self, x):
        return x[QUARTIER].map(self.par_quartier_).fillna(self.mediane_).to_numpy()


def compare_models(df: pd.DataFrame) -> dict:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x, y = df[features], df[TARGET]
    candidates = {
        "Reference : loyer median global": MedianeGlobale,
        "Reference : loyer median du quartier": MedianeParQuartier,
        "Regression lineaire (Ridge, log du loyer)": lambda: build_model_pipeline(Ridge(alpha=1.0)),
        "Gradient Boosting": lambda: build_model_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
        "Random Forest (modele retenu)": build_model_pipeline,
    }
    return {name: cross_validate_model(make_model, x, y) for name, make_model in candidates.items()}


def ablation(df: pd.DataFrame) -> dict:
    y = df[TARGET]
    num, cat = NUMERIC_FEATURES, CATEGORICAL_FEATURES
    sans = lambda columns, removed: [c for c in columns if c not in removed]
    variants = {
        "Modele retenu (chambres, type, standing, quartier, coordonnees)": (num, cat),
        "+ dates de publication (configuration d'origine)": (num + DATE_FEATURES, cat),
        "Sans coordonnees GPS (quartier seul)": (sans(num, ["Latitude", "Longitude"]), cat),
        "Sans nom de quartier (coordonnees seules)": (num, sans(cat, [QUARTIER])),
        "Sans aucune localisation": (sans(num, ["Latitude", "Longitude"]), sans(cat, [QUARTIER])),
        "Localisation seule": (["Latitude", "Longitude"], [QUARTIER]),
        "Sans standing": (num, sans(cat, ["Standing"])),
        "Sans type de logement": (num, sans(cat, ["Type logement"])),
        "Sans nombre de chambres": (sans(num, ["Chambres numeriques"]), cat),
    }
    results = {}
    for name, (numeric, categorical) in variants.items():
        make_model = lambda numeric=numeric, categorical=categorical: build_model_pipeline(
            numeric_features=numeric, categorical_features=categorical
        )
        results[name] = cross_validate_model(make_model, df[list(numeric) + list(categorical)], y)
    return results


def out_of_fold_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Predictions hors echantillon (moyennees sur les repetitions) et importance par permutation."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x, y = df[features], df[TARGET].astype(float).to_numpy()
    splitter = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    predictions = np.zeros(len(df))
    importances = []
    for train_index, test_index in splitter.split(x):
        model = build_model_pipeline().fit(x.iloc[train_index], y[train_index])
        predictions[test_index] += model.predict(x.iloc[test_index]) / CV_REPEATS
        permutation = permutation_importance(
            model,
            x.iloc[test_index],
            y[test_index],
            scoring="r2",
            n_repeats=5,
            random_state=RANDOM_STATE,
        )
        importances.append(permutation.importances_mean)
    importance = pd.DataFrame(importances, columns=features)
    importance_summary = {
        feature: {"baisse_r2": round(float(importance[feature].mean()), 4), "ecart_type": round(float(importance[feature].std(ddof=1)), 4)}
        for feature in importance.mean().sort_values(ascending=False).index
    }
    result = df.copy()
    result["Prix predit"] = predictions
    result["Erreur relative"] = np.abs(result["Prix predit"] - result[TARGET]) / result[TARGET]
    return result, importance_summary


def errors_by_group(oof: pd.DataFrame) -> dict:
    def summarize(group: pd.DataFrame) -> dict:
        return {
            "annonces": int(len(group)),
            "loyer_median": int(group[TARGET].median()),
            "mae": round(float((group["Prix predit"] - group[TARGET]).abs().mean())),
            "erreur_relative_mediane_pct": round(float(group["Erreur relative"].median() * 100), 1),
            "biais_moyen": round(float((group["Prix predit"] - group[TARGET]).mean())),
        }

    tranches = pd.cut(
        oof[TARGET],
        bins=[0, 300_000, 600_000, 1_000_000, 2_000_000, np.inf],
        labels=["< 300 000", "300 000 - 600 000", "600 000 - 1 000 000", "1 000 000 - 2 000 000", "> 2 000 000"],
    )
    par_tranche = {str(label): summarize(group) for label, group in oof.groupby(tranches, observed=True)}
    effectifs = oof[QUARTIER].value_counts()
    par_quartier = {quartier: summarize(oof[oof[QUARTIER] == quartier]) for quartier in effectifs.head(12).index}
    return {"par_tranche_de_loyer": par_tranche, "par_quartier": par_quartier}


def fmt(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def score_table(results: dict, first_column: str) -> list[str]:
    lines = [
        f"| {first_column} | MAE (F CFA) | RMSE (F CFA) | R2 | R2 (log) | Erreur relative mediane | Predictions a +/- 20 % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in results.items():
        lines.append(
            f"| {name} | {fmt(s['mae'])} | {fmt(s['rmse'])} | {s['r2']:.3f} (+/- {s['r2_ecart_type']:.3f}) | "
            f"{s['r2_log']:.3f} | {s['erreur_relative_mediane_pct']:.1f} % | {s['part_a_20pct']:.1f} % |"
        )
    return lines


def group_table(groups: dict, first_column: str) -> list[str]:
    lines = [
        f"| {first_column} | Annonces | Loyer median | MAE | Erreur relative mediane | Biais moyen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, g in groups.items():
        lines.append(
            f"| {name} | {g['annonces']} | {fmt(g['loyer_median'])} | {fmt(g['mae'])} | "
            f"{g['erreur_relative_mediane_pct']:.1f} % | {fmt(g['biais_moyen'])} |"
        )
    return lines


def write_report(report: dict) -> None:
    lines = [
        "# Evaluation du modele de prediction des loyers - appartements",
        "",
        f"Jeu de donnees : {report['annonces']} annonces, {report['quartiers']} quartiers, "
        f"loyer median {fmt(report['loyer_median'])} F CFA.",
        f"Protocole : validation croisee {CV_SPLITS} plis x {CV_REPEATS} repetitions ; "
        "chaque valeur est une moyenne sur les 15 plis, l'ecart-type du R2 est entre parentheses.",
        "",
        "## 1. Comparaison des modeles",
        "",
        *score_table(report["comparaison_modeles"], "Modele"),
        "",
        "## 2. Contribution des groupes de variables (Random Forest)",
        "",
        *score_table(report["ablation"], "Variables utilisees"),
        "",
        "## 3. Importance des variables par permutation",
        "",
        "Baisse moyenne du R2 hors echantillon quand on melange la variable.",
        "",
        "| Variable | Baisse du R2 | Ecart-type |",
        "|---|---:|---:|",
        *[f"| {name} | {v['baisse_r2']:.3f} | {v['ecart_type']:.3f} |" for name, v in report["importance_permutation"].items()],
        "",
        "## 4. Erreur selon le niveau de loyer",
        "",
        "Biais moyen positif = le modele surestime.",
        "",
        *group_table(report["erreurs"]["par_tranche_de_loyer"], "Tranche de loyer (F CFA)"),
        "",
        "## 5. Erreur dans les 12 quartiers les plus representes",
        "",
        *group_table(report["erreurs"]["par_quartier"], "Quartier"),
        "",
    ]
    EVALUATION_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_dataset(APPARTEMENTS_ML_CSV)
    oof, importance = out_of_fold_analysis(df)
    report = {
        "segment": "appartements",
        "annonces": int(len(df)),
        "quartiers": int(df[QUARTIER].nunique()),
        "loyer_median": int(df[TARGET].median()),
        "protocole": f"validation croisee {CV_SPLITS} plis x {CV_REPEATS} repetitions",
        "comparaison_modeles": compare_models(df),
        "ablation": ablation(df),
        "importance_permutation": importance,
        "erreurs": errors_by_group(oof),
    }
    ensure_metrics_dir()
    EVALUATION_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report)
    print(EVALUATION_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
