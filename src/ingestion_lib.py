import io
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

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
        return pd.read_csv(io.BytesIO(r.content), delimiter=";", encoding="cp1252")
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return None


def _guess_time_col(df: pd.DataFrame) -> str:
    for c in ["ReferenceTS", "reference_timestamp", "time"]:
        if c in df.columns:
            return c
    raise KeyError(f"Colonne temps introuvable. Colonnes dispo: {list(df.columns)}")


# ---- traitement : concat, parse temps, dédoublonnage, filtre 7 jours ----
def process_data(
    df_recent: pd.DataFrame,
    df_historical: pd.DataFrame,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
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


def build_output_path(output_dir: str, end_utc: datetime) -> Path:
    tz = ZoneInfo("Europe/Zurich")
    local_dt = end_utc.astimezone(tz)
    date_str = local_dt.strftime("%d%m%Y")  # 18112025
    filename = f"lastweek_{date_str}.csv"
    return Path(output_dir) / filename