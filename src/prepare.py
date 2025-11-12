import pandas as pd
import sys
import yaml
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

def main() -> None:
    if len(sys.argv) != 4:
        print("Arguments error. Usage:\n")
        print("\tpython3 prepare.py <raw-dataset-file> <raw-metadata-file> <prepared-dataset-folder>\n")
        exit(1)

    # Load parameters
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]
    separator_train_val = prepare_params["separator_train_val"]

    raw_dataset_file = Path(sys.argv[1])
    raw_metadata_file = Path(sys.argv[2])
    prepared_dataset_folder = Path(sys.argv[3])

    if not prepared_dataset_folder.exists():
        prepared_dataset_folder.mkdir(parents=True)

    # Read data
    meteo_df = pd.read_csv(raw_dataset_file, sep=",")

    metadata_df = pd.read_csv(raw_metadata_file, encoding='ISO-8859-1', sep=";",on_bad_lines='skip')
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
    feature_columns = meteo_df.drop(['station_abbr'], axis=1).columns
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
    ], axis=1)

    # Rename features
    features_df = features_df.rename({
        "Air temperature 2 m above ground; hourly mean": "air_temperature",
        "Relative air humidity 2 m above ground; hourly mean": "air_humidity",
        "Vapour pressure 2 m above ground; hourly mean": "vapour_pressure",
        "Atmospheric pressure at barometric altitude (QFE); hourly mean": "atmospheric_pressure",
        "Wind direction; hourly mean": "wind_direction",
        "Wind speed scalar; hourly mean in m/s": "wind_speed",
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

    print(features_df)
    imputer = KNNImputer(n_neighbors=3)
    features_df.loc[:] = imputer.fit_transform(features_df)

    meteo_df = pd.concat([
        meteo_df[['reference_timestamp', target_column]].rename({target_column: 'air_temperature'}, axis=1),
        features_df.shift(24).rename({col: col + " (lag 24)" for col in features_df.columns}, axis=1),
    ], axis=1)

    # Prepare test and train
    meteo_df = meteo_df.dropna(axis=0)
    meteo_df_train = meteo_df.iloc[:int(len(meteo_df)*separator_train_val)]
    meteo_df_val = meteo_df.iloc[int(len(meteo_df)*separator_train_val):]

    # Export to parquet
    for name, df in {"train": meteo_df_train, "test": meteo_df_val}.items():
        df.to_parquet(
            prepared_dataset_folder / f"{name}.parquet",
            index=False, engine="pyarrow", compression="snappy"
        )

if __name__ == "__main__":
    main()