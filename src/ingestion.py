#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from datetime import datetime, timedelta, timezone

from ingestion_lib import (
    download_csv,
    process_data,
    build_output_path,
)



def main():
    # --- args : RECENT_URL HISTORICAL_URL OUTPUT_CSV ---
    if len(sys.argv) != 4:
        print(
            "Usage: python fetch_meteoswiss.py RECENT_URL HISTORICAL_URL OUTPUT_CSV",
            file=sys.stderr,
        )
        sys.exit(2)

    recent_url = sys.argv[1]
    historical_url = sys.argv[2]
    output_dir = sys.argv[3]

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

    output_path = build_output_path(output_dir, end_utc)

    # Écriture CSV (crée le dossier au besoin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_last_week.to_csv(output_path, index=False)
    print(f"Écrit: {output_path} ({len(df_last_week)} lignes)")


if __name__ == "__main__":
    main()
