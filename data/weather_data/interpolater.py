import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from alive_progress import alive_bar

source_folder = 'weather/excel'
target_folder = 'weather_interpolated/parquet'

def interpolate_data(df):
    new_time_index = pd.date_range(start=df.index[0], end=df.index[-1], freq='5T')
    interpolated_df = pd.DataFrame(index=new_time_index)

    for column in df.columns:
        # Lineare-Interpolation
        if column in ['temperature_2m', 'pressure_msl', 'surface_pressure', 'wind_speed_10m', 'wind_speed_100m', 'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm', 'wind_direction_10m', 'wind_direction_100m', "wind_gusts_10m", 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'relative_humidity_2m']:
            f = interp1d(df.index.astype(np.int64), df[column], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolated_df[column] = f(interpolated_df.index.astype(np.int64))
        # Spline-Interpolation
        # elif column in ['wind_direction_10m', 'wind_direction_100m', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'relative_humidity_2m']:
        #     f = interp1d(df.index.astype(np.int64), df[column], kind='cubic', bounds_error=False, fill_value="extrapolate")
        #     interpolated_df[column] = f(interpolated_df.index.astype(np.int64))
        # Nearest-Neighbor-Interpolation
        elif column in ['rain', 'snowfall']:
            interpolated_df[column] = df[column].reindex(interpolated_df.index, method='nearest')
    
    return interpolated_df


excel_files = [f for f in os.listdir(source_folder) if f.endswith('.xlsx')]

with alive_bar(len(excel_files), bar="smooth", spinner="waves", length=85) as bar:

    for file_name in os.listdir(source_folder):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(source_folder, file_name)
            df = pd.read_excel(file_path, index_col='date')
            
            interpolated_df = interpolate_data(df)
            
            target_file_path = os.path.join(target_folder, file_name.replace('.xlsx', '.parquet'))
            interpolated_df.to_parquet(target_file_path)

            bar()
