from __future__ import annotations
import re
import unicodedata

import pandas as pd

from project_paths import NLP_LISTINGS_CSV


def normalize_text(value: str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_key(value: str) -> str:
    text = normalize_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("-", " ").replace("_", " ").replace("'", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


CORRECTIONS = {
    "almadies 2": "Almadies 2",
    "amitie": "Amitie",
    "cite biagui": "Cite Biagui",
    "cite damel": "Cite Damel",
    "cite keur gorgui": "Cite Keur Gorgui",
    "cite mixta": "Cite Mixta",
    "fenetre mermoz": "Mermoz",
    "grand yoff": "Grand Yoff",
    "hann bel air": "Hann Bel-Air",
    "hann marinas": "Hann Marinas",
    "hann maristes": "Hann Maristes",
    "hlm grand yoff": "HLM Grand Yoff",
    "liberte 1": "Liberte 1",
    "liberte 2": "Liberte 2",
    "liberte 3": "Liberte 3",
    "liberte 4": "Liberte 4",
    "liberte 5": "Liberte 5",
    "liberte 6": "Liberte 6",
    "liberte 6 extension": "Liberte 6 Extension",
    "ngor almadies": "Ngor-Almadies",
    "nord foire": "Nord Foire",
    "ouest foire": "Ouest Foire",
    "plateau": "Plateau",
    "point e": "Point E",
    "pointe e": "Point E",
    "sacre coeur": "Sacre-Coeur",
    "sacre coeur 1": "Sacre-Coeur 1",
    "sacre coeur 2": "Sacre-Coeur 2",
    "sacre coeur 3": "Sacre-Coeur 3",
    "sicap baobab": "Sicap Baobab",
    "sicap foire": "Sicap Foire",
    "sicap liberte": "Sicap Liberte",
    "sicap mbao": "Sicap Mbao",
    "sicap sacre coeur": "Sicap Sacre-Coeur",
    "sud foire": "Sud Foire",
    "vdn": "VDN",
    "zac mbao": "ZAC Mbao",
}


def nettoyer_quartier(value: str) -> str:
    key = canonical_key(value)
    if not key:
        return "N/A"
    if key in CORRECTIONS:
        return CORRECTIONS[key]
    if key.startswith("cite keur gorgui"):
        return "Cite Keur Gorgui"
    if key.startswith("ngor") and "almadies" in key:
        return "Ngor-Almadies"
    if key.startswith("almadies"):
        return "Almadies"
    if key.startswith("mermoz"):
        return "Mermoz"
    return normalize_text(value).title()


def main() -> None:
    df = pd.read_csv(NLP_LISTINGS_CSV, sep=";", encoding="utf-8-sig")
    df["Quartier nettoye"] = df["Lieu"].apply(nettoyer_quartier)
    df.to_csv(NLP_LISTINGS_CSV, sep=";", index=False, encoding="utf-8-sig")
    print(f"Quartiers normalises dans {NLP_LISTINGS_CSV.name}")


if __name__ == "__main__":
    main()

