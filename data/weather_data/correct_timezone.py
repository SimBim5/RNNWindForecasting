import pandas as pd
from timezonefinder import TimezoneFinder
from pytz import timezone
import os

# Schritt 1: Einlesen der Standortinformationen
# Pfad zur Excel-Tabelle mit Standortinformationen
locations_file = 'locations_wind_farms.xlsx'
locations_df = pd.read_excel(locations_file)

# Speichere die Koordinaten für jede Windfarm
wind_farm_locations = {row['Abbreviation']: (row['Latitude'], row['Longitude'])
                         for index, row in locations_df.iterrows()}

# Erstellen Sie ein TimezoneFinder-Objekt
tf = TimezoneFinder()

# Zeitzone für jede Windfarm ermitteln
wind_farm_timezones = {
    name: tf.timezone_at(lat=lat, lng=lng) for name, (lat, lng) in wind_farm_locations.items()
}

# Funktion zum Konvertieren der Zeitstempel
def convert_timestamps(df, timezone_name):
    tz = timezone(timezone_name)
    df['date'] = df['date'].dt.tz_localize(None)
    return df

# Ordnerpfade
excel_folder = 'weather_wind_farms\excel'
parquet_folder = 'weather_wind_farms\parquet'
interpolated_parquet_folder = 'weather_wind_farms_interpolated'

for wind_farm in wind_farm_locations:
    excel_file = os.path.join(excel_folder, wind_farm + '.xlsx')
    parquet_file = os.path.join(parquet_folder, wind_farm + '.parquet')
    
    df = pd.read_excel(excel_file)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert(wind_farm_timezones[wind_farm])
    df['date'] = df['date'].dt.tz_localize(None)  # Entferne die Zeitzone
    df.to_excel(excel_file, index=False)  # Speichere die Datei ohne Zeitzone

    df = pd.read_parquet(parquet_file)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert(wind_farm_timezones[wind_farm])
    df['date'] = df['date'].dt.tz_localize(None)  # Entferne die Zeitzone
    df.to_parquet(parquet_file)

    df = pd.read_parquet(interpolated_parquet_folder)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert(wind_farm_timezones[wind_farm])
    df['date'] = df['date'].dt.tz_localize(None)  # Entferne die Zeitzone
    df.to_parquet(interpolated_parquet_folder)