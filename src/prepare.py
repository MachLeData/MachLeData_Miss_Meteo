import io
import sys
import yaml
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

# -----------------------------
# Téléchargement + assemblage API
# -----------------------------

def download_csv(url: str) -> Optional[pd.DataFrame]:
    """Télécharge un CSV (souvent ';' + cp1252) et renvoie un DataFrame."""
    try:
        hdrs = {
            "User-Agent": "Mozilla/5.0 (compatible; MeteoDataFetcher/1.0)",
            "Accept": "text/csv, */*;q=0.1",
        }
        r = requests.get(url, headers=hdrs, timeout=60)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), delimiter=';', encoding="cp1252")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def _guess_time_col(df: pd.DataFrame) -> str:
    for c in ("reference_timestamp", "ReferenceTS", "time", "timestamp"):
        if c in df.columns:
            return c
    raise KeyError(f"Colonne temps introuvable. Colonnes dispo: {list(df.columns)}")


def load_dataset_from_api(arg_urls: str) -> pd.DataFrame:
    """Charge 2 URLs séparées par ';' (historical;recent/now), concatène,
    standardise la colonne temporelle en `reference_timestamp` et dédoublonne.
    """
    if ";" not in arg_urls or ("http" not in arg_urls and "https" not in arg_urls):
        raise ValueError(
            "Le premier argument doit être deux URLs séparées par ';' (historical;recent)."
        )

    historical_url, recent_url = [u.strip() for u in arg_urls.split(";")[:2]]
    df_hist = download_csv(historical_url)
    df_recent = download_csv(recent_url)

    if df_hist is None or df_recent is None:
        raise RuntimeError("Téléchargement API échoué (historical ou recent manquant).")

    df = pd.concat([df_hist, df_recent], ignore_index=True)
    time_col = _guess_time_col(df)
    df = df.rename(columns={time_col: "reference_timestamp"})

    # Parsing robuste (souvent dd.mm.yyyy HH:MM)
    df["reference_timestamp"] = pd.to_datetime(
        df["reference_timestamp"], errors="coerce", dayfirst=True
    )

    df = (
        df.dropna(subset=["reference_timestamp"]).sort_values("reference_timestamp")
          .drop_duplicates(subset=["reference_timestamp"], keep="last")
          .reset_index(drop=True)
    )
    return df


# -----------------------------
# Script principal (pipeline inchangé)
# -----------------------------

def main() -> None:
    if len(sys.argv) != 4:
        print("Arguments error. Usage:")
        print("	python3 prepare.py \"<URL_historical>;<URL_recent>\" <raw-metadata-file> <prepared-dataset-folder>")
        sys.exit(1)

    # Paramètres
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]
    separator_train_val = prepare_params["separator_train_val"]

    api_urls = sys.argv[1]                 # "historical;recent"
    raw_metadata_file = Path(sys.argv[2])
    prepared_dataset_folder = Path(sys.argv[3])

    prepared_dataset_folder.mkdir(parents=True, exist_ok=True)

    # --- Read data depuis API ---
    meteo_df = load_dataset_from_api(api_urls)

    metadata_df = pd.read_csv(raw_metadata_file, encoding='ISO-8859-1', sep=';', on_bad_lines='skip')
    metadata_df = metadata_df.set_index("parameter_shortname")

    # Renommage abréviations -> descriptions
    meteo_df = meteo_df.rename({
        col: metadata_df.loc[col]["parameter_description_en"]
        for col in meteo_df.columns if col in metadata_df.index
    }, axis=1)

    # Standardiser le datetime
    if "reference_timestamp" not in meteo_df.columns:
        time_col = _guess_time_col(meteo_df)
        meteo_df = meteo_df.rename(columns={time_col: "reference_timestamp"})

    meteo_df['reference_timestamp'] = pd.to_datetime(
        meteo_df['reference_timestamp'], dayfirst=True, errors='coerce'
    )
    meteo_df = meteo_df.dropna(subset=['reference_timestamp']).sort_values('reference_timestamp')

    # Suppression colonnes vides
    meteo_df = meteo_df.dropna(axis=1, how='all')

    # Cible + features
    target_column = 'Air temperature 2 m above ground; hourly mean'

    # 'station_abbr' optionnelle
    base_for_features = meteo_df.drop(columns=['station_abbr'], errors='ignore')
    feature_columns = base_for_features.columns
    features_df = meteo_df[feature_columns].copy()

    # Colonnes redondantes (inchangé)
    to_drop = [
        'Air temperature 2 m above ground; hourly minimum',
        'Air temperature 2 m above ground; hourly maximum',
        'Dew point 2 m above ground; hourly mean',
        'Pressure reduced to sea level according to standard atmosphere (QNH); hourly mean',
        'Pressure reduced to sea level (QFF); hourly mean',
        'Gust peak (one second); hourly maximum in m/s',
        'Gust peak (one second); hourly maximum in km/h',
        'Gust peak (three seconds); hourly maximum in m/s',
        'Gust peak (three seconds); hourly maximum in km/h',
        'Wind speed scalar; hourly mean in km/h',
        'Reference evaporation from FAO; hourly total',
        'Sunshine duration; hourly total',
    ]
    features_df = features_df.drop(columns=[c for c in to_drop if c in features_df.columns], errors='ignore')

    # Renommer features (inchangé)
    features_df = features_df.rename({
        "Air temperature 2 m above ground; hourly mean": "air_temperature",
        "Relative air humidity 2 m above ground; hourly mean": "air_humidity",
        "Vapour pressure 2 m above ground; hourly mean": "vapour_pressure",
        "Atmospheric pressure at barometric altitude (QFE); hourly mean": "atmospheric_pressure",
        "Wind direction; hourly mean": "wind_direction",
        "Wind speed scalar; hourly mean in m/s": "wind_speed",
        "Precipitation; hourly total": "precipitation",
        "Global radiation; hourly mean": "global_radiation",
        "Diffuse radiation; hourly mean": "diffuse_radiation",
        "reference_timestamp": "reference_timestamp",
    }, axis=1)

    # Transformations
    def sin_transformer(period):
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

    def cos_transformer(period):
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    if "reference_timestamp" not in features_df.columns:
        features_df = features_df.join(meteo_df["reference_timestamp"])  # sécurité

    features_df['hour'] = meteo_df["reference_timestamp"].dt.hour
    features_df['day'] = meteo_df["reference_timestamp"].dt.day
    features_df['month'] = meteo_df["reference_timestamp"].dt.month

    if "reference_timestamp" in features_df.columns:
        features_df = features_df.drop("reference_timestamp", axis=1)

    column_transformer = ColumnTransformer(
        transformers=[
            ("wind_direction_sin", sin_transformer(360), ["wind_direction"]),
            ("wind_direction_cos", cos_transformer(360), ["wind_direction"]),
            ("day_sin", sin_transformer(24), ["hour"]),
            ("day_cos", cos_transformer(24), ["hour"]),
            ("month_sin", sin_transformer(30), ["day"]),
            ("month_cos", cos_transformer(30), ["day"]),
            ("year_sin", sin_transformer(12), ["month"]),
            ("year_cos", cos_transformer(12), ["month"]),
            ("precipitation", PowerTransformer(), ["precipitation"]),
            ("global_radiation", PowerTransformer(), ["global_radiation"]),
            ("diffuse_radiation", PowerTransformer(), ["diffuse_radiation"]),
        ],
        remainder=StandardScaler(),
    )

    column_transformer.set_output(transform="pandas")
    features_df = column_transformer.fit_transform(features_df)

    def transform_column_name(col: str) -> str:
        splitted = col.split("__")
        col = splitted[1]
        if "sin" in splitted[0]:
            col += "_sin"
        elif "cos" in splitted[0]:
            col += "_cos"
        return col

    features_df = features_df.rename({col: transform_column_name(col) for col in features_df.columns}, axis=1)

    print(features_df)

    # Imputation
    imputer = KNNImputer(n_neighbors=3)
    features_df.loc[:] = imputer.fit_transform(features_df)

    # Table finale: cible + features laggées de 24h
    if target_column not in meteo_df.columns:
        raise KeyError(f"Colonne cible manquante: '{target_column}'. Vérifiez la concordance metadata/dataset.")

    final_df = pd.concat([
        meteo_df[['reference_timestamp', target_column]].rename({target_column: 'air_temperature'}, axis=1),
        features_df.shift(24).rename({col: col + " (lag 24)" for col in features_df.columns}, axis=1),
    ], axis=1)

    # Split
    final_df = final_df.dropna(axis=0)
    split_idx = int(len(final_df) * separator_train_val)
    meteo_df_train = final_df.iloc[:split_idx]
    meteo_df_val = final_df.iloc[split_idx:]

    # Export
    for name, df in {"train": meteo_df_train, "test": meteo_df_val}.items():
        (prepared_dataset_folder / f"{name}.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            prepared_dataset_folder / f"{name}.parquet",
            index=False, engine="pyarrow", compression="snappy"
        )


if __name__ == "__main__":
    main()
