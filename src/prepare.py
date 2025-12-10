# prepare.py

import sys
from pathlib import Path

from prepare_lib import (
    load_prepare_params,
    load_and_merge_meteo,
    load_metadata,
    apply_metadata_to_meteo,
    build_raw_feature_matrix,
    preprocess_features,
    build_supervised_dataframe,
    split_train_val_by_historical,
    save_splits_to_parquet,
)


def main() -> None:
    if len(sys.argv) != 5:
        print("Arguments error. Usage:\n")
        print(
            "\tpython3 prepare.py "
            "<raw-dataset-file> "
            "<finetuning-dataset-folder> "
            "<raw-metadata-file> "
            "<prepared-dataset-folder>\n"
        )
        exit(1)

    raw_dataset_file = Path(sys.argv[1])
    finetuning_dataset_folder = Path(sys.argv[2])
    raw_metadata_file = Path(sys.argv[3])
    prepared_dataset_folder = Path(sys.argv[4])

    # 1) paramètres
    prepare_params = load_prepare_params("params.yaml")
    separator_train_val = prepare_params["separator_train_val"]

    # 2) chargement + fusion des données
    meteo_df = load_and_merge_meteo(raw_dataset_file, finetuning_dataset_folder)
    metadata_df = load_metadata(raw_metadata_file)
    meteo_df = apply_metadata_to_meteo(meteo_df, metadata_df)

    # 3) features
    features_df = build_raw_feature_matrix(meteo_df)
    features_df = preprocess_features(features_df)

    # 4) dataset supervisé (lag 24)
    supervised_df = build_supervised_dataframe(meteo_df, features_df)

    # 5) splits + sauvegarde
    splits = split_train_val_by_historical(supervised_df, separator_train_val)
    save_splits_to_parquet(splits, prepared_dataset_folder)


if __name__ == "__main__":
    main()
