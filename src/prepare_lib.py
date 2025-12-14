from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

TARGET_COLUMN = "Air temperature 2 m above ground; hourly mean"

REDUNDANT_FEATURES = [
    "Air temperature 2 m above ground; hourly minimum",
    "Air temperature 2 m above ground; hourly maximum",
    "Dew point 2 m above ground; hourly mean",
    "Pressure reduced to sea level according to standard atmosphere (QNH); hourly mean",
    "Pressure reduced to sea level (QFF); hourly mean",
    "Gust peak (one second); hourly maximum in m/s",
    "Gust peak (one second); hourly maximum in km/h",
    "Gust peak (three seconds); hourly maximum in m/s",
    "Gust peak (three seconds); hourly maximum in km/h",
    "Wind speed scalar; hourly mean in km/h",
    "Reference evaporation from FAO; hourly total",
    "Sunshine duration; hourly total",
]

FEATURE_RENAME_MAP = {
    "Air temperature 2 m above ground; hourly mean": "air_temperature",
    "Relative air humidity 2 m above ground; hourly mean": "air_humidity",
    "Vapour pressure 2 m above ground; hourly mean": "vapour_pressure",
    "Atmospheric pressure at barometric altitude (QFE); hourly mean": "atmospheric_pressure",
    "Wind direction; hourly mean": "wind_direction",
    "Wind speed scalar; hourly mean in m/s": "wind_speed",
    "Precipitation; hourly total": "precipitation",
    "Global radiation; hourly mean": "global_radiation",
    "Diffuse radiation; hourly mean": "diffuse_radiation",
}

def load_prepare_params(params_path: str | Path = "params.yaml") -> Dict:
    params_path = Path(params_path)
    with open(params_path, "r") as f:
        all_params = yaml.safe_load(f)
    return all_params["prepare"]


def pick_latest_lastweek_csv(folder: Path) -> Path:
    if not folder.exists() or not folder.is_dir():
        print(f"Erreur: le dossier {folder} n'existe pas ou n'est pas un dossier.")
        exit(1)

    candidates = list(folder.glob("lastweek_*.csv"))
    if not candidates:
        print(f"Aucun fichier lastweek_*.csv trouvé dans {folder}")
        exit(1)

    def parse_date_from_name(path: Path) -> datetime:
        # nom sans extension, ex: "lastweek_18112025"
        stem = path.stem
        date_str = stem.split("lastweek_")[-1]
        try:
            return datetime.strptime(date_str, "%d%m%Y")
        except ValueError:
            return datetime.min

    latest_file = max(candidates, key=parse_date_from_name)
    print(f"Utilisation du fichier météo : {latest_file}")
    return latest_file


def load_and_merge_meteo(
    raw_dataset_file: Path,
    finetuning_dataset_folder: Path,
) -> pd.DataFrame:
    historical_meteo_df = pd.read_csv(raw_dataset_file, sep=";")
    historical_meteo_df["historical"] = True

    finetuning_dataset_file = pick_latest_lastweek_csv(finetuning_dataset_folder)
    meteo_df = pd.read_csv(finetuning_dataset_file, sep=",")
    meteo_df["historical"] = False

    meteo_df = pd.concat([meteo_df, historical_meteo_df], axis=0)
    return meteo_df


def load_metadata(raw_metadata_file: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(
        raw_metadata_file,
        encoding="ISO-8859-1",
        sep=";",
        on_bad_lines="skip",
    )
    metadata_df = metadata_df.set_index("parameter_shortname")
    return metadata_df


def apply_metadata_to_meteo(
    meteo_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    rename_map = {
        col: metadata_df.loc[col]["parameter_description_en"]
        for col in meteo_df.columns
        if col in metadata_df.index
    }

    meteo_df = meteo_df.rename(rename_map, axis=1)
    meteo_df["reference_timestamp"] = pd.to_datetime(
        meteo_df["reference_timestamp"],
        dayfirst=True,
    )

    meteo_df = meteo_df.dropna(axis=1, how="all")
    return meteo_df

def build_raw_feature_matrix(meteo_df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = meteo_df.drop(["station_abbr", "historical"], axis=1).columns
    features_df = meteo_df[feature_columns].copy()

    # Remove highly similar features
    features_df = features_df.drop(REDUNDANT_FEATURES, axis=1, errors="ignore")

    # Rename features
    features_df = features_df.rename(FEATURE_RENAME_MAP, axis=1)

    # Time features
    features_df["hour"] = meteo_df["reference_timestamp"].dt.hour
    features_df["day"] = meteo_df["reference_timestamp"].dt.day
    features_df["month"] = meteo_df["reference_timestamp"].dt.month

    # Supprimer les colonnes non numériques susceptibles de rester
    features_df = features_df.drop(
        columns=["reference_timestamp", "ts_utc", "ts_local"],
        errors="ignore",
    )

    return features_df


def sin_transformer(period: float) -> FunctionTransformer:
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period: float) -> FunctionTransformer:
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def create_column_transformer() -> ColumnTransformer:
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
    return column_transformer


def transform_column_name(col: str) -> str:
    parts = col.split("__", 1)
    if len(parts) == 2:
        prefix, base = parts
        if "sin" in prefix:
            return f"{base}_sin"
        if "cos" in prefix:
            return f"{base}_cos"
        return base
    return col


def impute_missing_values(
    features_df: pd.DataFrame,
    n_neighbors: int = 3,
) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    features_df.loc[:] = imputer.fit_transform(features_df)
    return features_df


def preprocess_features(features_df: pd.DataFrame, df_historical: pd.DataFrame = None) -> pd.DataFrame:
    # Renommer les colonnes
    features_df = features_df.rename(
        {col: transform_column_name(col) for col in features_df.columns},
        axis=1,
    )

    df_historical = df_historical.rename(
        {col: transform_column_name(col) for col in df_historical.columns},
        axis=1,
    )

    if df_historical is not None:
        column_transformer = create_column_transformer().fit(df_historical)
        features_df = column_transformer.transform(features_df)
    else:
        column_transformer = create_column_transformer()
        features_df = column_transformer.fit_transform(features_df)

    # Imputation
    features_df = impute_missing_values(features_df, n_neighbors=3)
    return features_df


# ======================================================================
# Dataset supervisé + splits
# ======================================================================

def build_supervised_dataframe(
    meteo_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    lag_hours: int = 24,
) -> pd.DataFrame:
    historical = meteo_df["historical"].copy()

    supervised_df = pd.concat(
        [
            meteo_df[["reference_timestamp", target_column]].rename(
                {target_column: "air_temperature"},
                axis=1,
            ),
            features_df.shift(lag_hours).rename(
                {col: f"{col} (lag {lag_hours})" for col in features_df.columns},
                axis=1,
            ),
        ],
        axis=1,
    )

    supervised_df["historical"] = historical
    supervised_df = supervised_df.dropna(axis=0)

    return supervised_df


def split_train_val_by_historical(
    meteo_df: pd.DataFrame,
    separator_train_val: float,
) -> Dict[str, pd.DataFrame]:
    historical_df = meteo_df[meteo_df["historical"]]
    finetuning_df = meteo_df[~meteo_df["historical"]]

    def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * separator_train_val)
        return df.iloc[:split_idx], df.iloc[split_idx:]

    historical_train, historical_val = split(historical_df)
    finetuning_train, finetuning_val = split(finetuning_df)

    return {
        "train_historical": historical_train,
        "val_historical": historical_val,
        "train_finetuning": finetuning_train,
        "val_finetuning": finetuning_val,
    }


def save_splits_to_parquet(
    splits: Dict[str, pd.DataFrame],
    output_folder: Path,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        df.to_parquet(
            output_folder / f"{name}.parquet",
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
