import pandas as pd
import sys
import yaml
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from datetime import datetime


def pick_latest_lastweek_csv(folder: Path) -> Path:
    """
    Cherche les fichiers lastweek_*.csv dans `folder` et renvoie
    celui avec la date la plus récente dans le nom.
    Ex: lastweek_17112025.csv, lastweek_18112025.csv -> prend 18112025.
    """
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

def main() -> None:
    if len(sys.argv) != 5:
        print("Arguments error. Usage:\n")
        print("\tpython3 prepare.py <raw-dataset-file> <finetuning-dataset-folder> <raw-metadata-file> <prepared-dataset-folder>\n")
        exit(1)

    # Load parameters
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]
    separator_train_val = prepare_params["separator_train_val"]

    raw_dataset_file = Path(sys.argv[1])
    finetuning_dataset_folder = Path(sys.argv[2])
    raw_metadata_file = Path(sys.argv[3])
    prepared_dataset_folder = Path(sys.argv[4])

    if not prepared_dataset_folder.exists():
        prepared_dataset_folder.mkdir(parents=True)

    # Read data
    historical_meteo_df = pd.read_csv(raw_dataset_file, sep=";")
    historical_meteo_df['historical'] = True

    finetuning_dataset_file = pick_latest_lastweek_csv(finetuning_dataset_folder)
    meteo_df = pd.read_csv(finetuning_dataset_file, sep=",")

    # Combine historical and fine-tuning files
    meteo_df['historical'] = False

    meteo_df = pd.concat([
        meteo_df,
        historical_meteo_df
    ], axis=0)

    metadata_df = pd.read_csv(raw_metadata_file, encoding='ISO-8859-1', sep=";", on_bad_lines='skip')
    metadata_df = metadata_df.set_index("parameter_shortname")

    # Rename columns abbreviation to parameter description
    meteo_df = meteo_df.rename({
        col: metadata_df.loc[col]["parameter_description_en"]
        for col in
        meteo_df.columns
        if col in metadata_df.index
    }, axis=1)
    meteo_df['reference_timestamp'] = pd.to_datetime(meteo_df['reference_timestamp'], dayfirst=True)

    # Remove columns without any data
    meteo_df = meteo_df.dropna(axis=1, how='all')

    # Prepare data
    target_column = 'Air temperature 2 m above ground; hourly mean'
    feature_columns = meteo_df.drop(['station_abbr', 'historical'], axis=1).columns

    features_df = meteo_df[feature_columns]

    # Remove highly similar features
    features_df = features_df.drop([
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
        'Wind direction; hourly mean',
        "Wind speed scalar; hourly mean in m/s",
    ], axis=1)

    # Rename features
    features_df = features_df.rename({
        "Air temperature 2 m above ground; hourly mean": "air_temperature",
        "Relative air humidity 2 m above ground; hourly mean": "air_humidity",
        "Vapour pressure 2 m above ground; hourly mean": "vapour_pressure",
        "Atmospheric pressure at barometric altitude (QFE); hourly mean": "atmospheric_pressure",
        "Precipitation; hourly total": "precipitation",
        "Global radiation; hourly mean": "global_radiation",
        "Diffuse radiation; hourly mean": "diffuse_radiation"
    }, axis=1)

    # Preprocess features

    def sin_transformer(period):
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

    def cos_transformer(period):
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    features_df['hour'] = meteo_df["reference_timestamp"].dt.hour
    features_df['day'] = meteo_df["reference_timestamp"].dt.day
    features_df['month'] = meteo_df["reference_timestamp"].dt.month

    # supprime les colonnes non numériques susceptibles de rester
    features_df = features_df.drop(columns=['reference_timestamp', 'ts_utc', 'ts_local'], errors='ignore')

    column_transformer = ColumnTransformer(
        transformers=[
            # ("wind_direction_sin", sin_transformer(360), ["wind_direction"]),
            # ("wind_direction_cos", cos_transformer(360), ["wind_direction"]),
            ("day_sin", sin_transformer(24), ["hour"]),
            ("day_cos", cos_transformer(24), ["hour"]),
            ("month_sin", sin_transformer(30), ["day"]),
            ("month_cos", cos_transformer(30), ["day"]),
            ("year_sin", sin_transformer(12), ["month"]),
            ("year_cos", cos_transformer(12), ["month"]),
            ("precipitation", PowerTransformer(), ["precipitation"]),
            ("global_radiation", PowerTransformer(), ["global_radiation"]),
            # ("diffuse_radiation", PowerTransformer(), ["diffuse_radiation"]),
        ],
        remainder=StandardScaler(),
    )

    column_transformer.set_output(transform="pandas")

    features_df = column_transformer.fit_transform(features_df)

    def transform_column_name(col):
        parts = col.split("__", 1)
        if len(parts) == 2:
            prefix, base = parts
            if "sin" in prefix:
                return f"{base}_sin"
            if "cos" in prefix:
                return f"{base}_cos"
            return base
        return col  # colonnes du remainder gardent leur nom

    # Fix columns names
    features_df = features_df.rename({col: transform_column_name(col) for col in features_df.columns}, axis=1)
    imputer = KNNImputer(n_neighbors=3) # Maybe a data leak here because we do it before splitting
    features_df.loc[:] = imputer.fit_transform(features_df)

    historical = meteo_df['historical']

    meteo_df = pd.concat([
        meteo_df[['reference_timestamp', target_column]].rename({target_column: 'air_temperature'}, axis=1),
        features_df.shift(24).rename({col: col + " (lag 24)" for col in features_df.columns}, axis=1),
    ], axis=1)

    # Put back the historical column
    meteo_df['historical'] = historical

    # Prepare test and train
    meteo_df = meteo_df.dropna(axis=0)

    # Separate historical/finetuning
    historical_meteo_df = meteo_df[meteo_df['historical']]
    finetuning_meteo_df = meteo_df[~meteo_df['historical']]

    historical_meteo_df_train = historical_meteo_df.iloc[:int(len(historical_meteo_df)*separator_train_val)]
    historical_meteo_df_val = historical_meteo_df.iloc[int(len(historical_meteo_df)*separator_train_val):]

    finetuning_meteo_df_train = finetuning_meteo_df.iloc[:int(len(finetuning_meteo_df)*separator_train_val)]
    finetuning_meteo_df_val = finetuning_meteo_df.iloc[int(len(finetuning_meteo_df)*separator_train_val):]    

    # Export to parquet
    for name, df in {
        "train_historical": historical_meteo_df_train,
        "val_historical": historical_meteo_df_val,
        "train_finetuning": finetuning_meteo_df_train,
        "val_finetuning": finetuning_meteo_df_val,        
    }.items():
        df.to_parquet(
            prepared_dataset_folder / f"{name}.parquet",
            index=False, engine="pyarrow", compression="snappy"
        )

if __name__ == "__main__":
    main()