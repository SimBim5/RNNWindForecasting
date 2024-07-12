import os
import pandas as pd
from alive_progress import alive_bar

source_folder = 'weather_wind_farms/excel'  # Ordner mit den Excel-Dateien
target_folder = 'weather_wind_farms/parquet'  # Zielordner für die Parquet-Dateien

excel_files = [f for f in os.listdir(source_folder) if f.endswith('.xlsx')]

with alive_bar(len(excel_files), bar="smooth", spinner="waves", length=85) as bar:
    for file_name in excel_files:
        file_path = os.path.join(source_folder, file_name)
        df = pd.read_excel(file_path)

        target_file_path = os.path.join(target_folder, file_name.replace('.xlsx', '.parquet'))
        df.to_parquet(target_file_path)

        bar()

print("Konvertierung abgeschlossen.")
