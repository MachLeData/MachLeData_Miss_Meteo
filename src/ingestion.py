#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import io
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---- téléchargement d'un CSV MeteoSwiss ----
def download_csv(url: str) -> pd.DataFrame | None:
    """Télécharge un CSV MeteoSwiss et renvoie un DataFrame pandas."""
    try:
        hdrs = {
            "User-Agent": "Mozilla/5.0 (compatible; MeteoDataFetcher/1.0)",
            "Accept": "text/csv, */*;q=0.1",
        }
        r = requests.get(url, headers=hdrs, timeout=30)
        r.raise_for_status()
        # CSV MeteoSwiss: séparateur ';', encodage Windows-1252
        return pd.read_csv(io.BytesIO(r.content), delimiter=';', encoding="cp1252")
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return None

def _guess_time_col(df: pd.DataFrame) -> str:
    for c in ["ReferenceTS", "reference_timestamp", "time"]:
        if c in df.columns:
            return c
    raise KeyError(f"Colonne temps introuvable. Colonnes dispo: {list(df.columns)}")

# ---- traitement : concat, parse temps, dédoublonnage, filtre 7 jours ----
def process_data(df_recent: pd.DataFrame, df_historical: pd.DataFrame,
                 start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    if df_recent is None or df_historical is None:
        raise ValueError("df_recent/df_historical manquant")

    # 1) concat
    df = pd.concat([df_historical, df_recent], ignore_index=True)

    # 2) parse du timestamp (UTC, format 'dd.mm.yyyy HH:MM')
    time_col = _guess_time_col(df)
    df["ts_utc"] = pd.to_datetime(
        df[time_col], format="%d.%m.%Y %H:%M", utc=True, errors="coerce"
    )
    df = df.dropna(subset=["ts_utc"])

    # 3) tri + dédoublonnage sur le timestamp
    df = df.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")

    # 4) filtre dernière semaine
    df = df[(df["ts_utc"] > start_utc) & (df["ts_utc"] <= end_utc)].copy()
    return df.reset_index(drop=True)

def main():
    # --- args : RECENT_URL HISTORICAL_URL OUTPUT_CSV ---
    if len(sys.argv) != 4:
        print("Usage: python fetch_meteoswiss.py RECENT_URL HISTORICAL_URL OUTPUT_CSV", file=sys.stderr)
        sys.exit(2)

    recent_url = sys.argv[1]
    historical_url = sys.argv[2]
    output_csv = sys.argv[3]

    # ---- fenêtre : dernière semaine glissante ----
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=7)

    # ---- exécution (même flow que le notebook) ----
    df_recent = download_csv(recent_url)
    df_historical = download_csv(historical_url)

    if df_recent is None or df_historical is None:
        sys.exit(1)

    try:
        df_last_week = process_data(df_recent, df_historical, start_utc, end_utc)
    except Exception as e:
        print(f"Processing error: {e}", file=sys.stderr)
        sys.exit(1)

    # Affichages similaires au notebook
    print(df_last_week.head())
    print(df_last_week.tail())

    # Écriture CSV (crée le dossier au besoin)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_last_week.to_csv(output_csv, index=False)
    print(f"Écrit: {output_csv} ({len(df_last_week)} lignes)")

if __name__ == "__main__":
    main()
