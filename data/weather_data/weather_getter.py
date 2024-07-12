import os
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import time


def save_to_excel(data_df, coordinates_df, filename, data_sheet_name, coord_sheet_name):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        data_df.to_excel(writer, sheet_name=data_sheet_name, index=False)
        coordinates_df.to_excel(writer, sheet_name=coord_sheet_name, index=False)

def make_api_call_with_retry(url, params, max_retries=5, retry_delay=60):
    for attempt in range(max_retries):
        try:
            responses = openmeteo.weather_api(url, params=params)
            return responses
        except Exception as e:
            if 'Minutely API request limit exceeded' in str(e):
                print(f"API request limit exceeded, attempt {attempt + 1} of {max_retries}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait for retry_delay seconds before retrying
            else:
                print(f"An error occurred: {e}")
                break
    raise Exception("API call failed after maximum number of retries.")

# Load the Excel file
locations_df = pd.read_excel('locations_wind_farms.xlsx')

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Create a new directory for the Excel files
output_dir = 'weather_wind_farms\excel'
os.makedirs(output_dir, exist_ok=True)

# Filter out locations for which an Excel file already exists
existing_files = {file.split('.')[0] for file in os.listdir(output_dir)}
locations_df = locations_df[~locations_df['Abbreviation'].isin(existing_files)]

# API parameters that are constant for all requests
url = "https://customer-archive-api.open-meteo.com/v1/archive"
start_date = "2010-01-01"
end_date = "2023-12-31"
variables = ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", 
             "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", 
             "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "wind_speed_100m", 
             "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", 
             "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", 
             "soil_moisture_100_to_255cm"]

# Iterate over each wind farm location
for index, row in locations_df.iterrows():

    abbreviation = row['Abbreviation']
    latitude = row['Latitude']
    longitude = row['Longitude']

    print(f"Fetching data for {abbreviation} at Coordinates {latitude}°E {longitude}°N")

    # Define parameters for the API call
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
	    "apikey": "BRVyp47vzYMHgEK8",
        "timezone": "auto"
    }
    
    # Make the API call
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
    print(f"Elevation {response.Elevation()} m asl")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
    hourly_pressure_msl = hourly.Variables(4).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(5).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(7).ValuesAsNumpy()
    hourly_cloud_cover_mid = hourly.Variables(8).ValuesAsNumpy()
    hourly_cloud_cover_high = hourly.Variables(9).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(10).ValuesAsNumpy()
    hourly_wind_speed_100m = hourly.Variables(11).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(12).ValuesAsNumpy()
    hourly_wind_direction_100m = hourly.Variables(13).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(14).ValuesAsNumpy()
    hourly_soil_moisture_0_to_7cm = hourly.Variables(15).ValuesAsNumpy()
    hourly_soil_moisture_7_to_28cm = hourly.Variables(16).ValuesAsNumpy()
    hourly_soil_moisture_28_to_100cm = hourly.Variables(17).ValuesAsNumpy()
    hourly_soil_moisture_100_to_255cm = hourly.Variables(18).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s"),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["rain"] = hourly_rain
    hourly_data["snowfall"] = hourly_snowfall
    hourly_data["pressure_msl"] = hourly_pressure_msl
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
    hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
    hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
    hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
    hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm
    hourly_data["soil_moisture_7_to_28cm"] = hourly_soil_moisture_7_to_28cm
    hourly_data["soil_moisture_28_to_100cm"] = hourly_soil_moisture_28_to_100cm
    hourly_data["soil_moisture_100_to_255cm"] = hourly_soil_moisture_100_to_255cm

    hourly_dataframe = pd.DataFrame(data = hourly_data)

    coordinates_data = {
        'Original Latitude': [latitude],
        'Original Longitude': [longitude],
        'Response Latitude': [response.Latitude()],
        'Response Longitude': [response.Longitude()]
    }
    coordinates_df = pd.DataFrame(coordinates_data)

    hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])
    offset = pd.Timestamp('2010-01-01 00:00:00') - hourly_dataframe['date'].iloc[0]
    hourly_dataframe['date'] += offset

    # Define the Excel file path
    excel_filename = os.path.join(output_dir, f"{abbreviation}.xlsx")
    # Save the DataFrame with coordinates in the first line
    save_to_excel(hourly_dataframe, coordinates_df, excel_filename, abbreviation, 'coordinates')

    print(f"Data for {abbreviation} saved to {excel_filename}")