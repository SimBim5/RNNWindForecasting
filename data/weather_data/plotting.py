import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths to the folders
interpolated_folder = 'weather_interpolated/parquet'
original_folder = 'weather/parquet'
specific_date = '2023-01-01'

# Assuming you have a specific way to list files that match your criteria
# Get a list of columns from one of the files (assuming all files have the same structure)
if os.listdir(interpolated_folder):
    sample_file = os.listdir(interpolated_folder)[0]
    sample_df = pd.read_parquet(os.path.join(interpolated_folder, sample_file))
    columns = sample_df.columns
    print("Available columns to plot:")
    for i, column in enumerate(columns, 1):
        print(f"{i}. {column}")

    # Let the user select which column to plot
    selected_column_index = int(input("Enter the number of the column you want to plot: ")) - 1
    selected_column = columns[selected_column_index]

    for interpolated_file_name in os.listdir(interpolated_folder):
        if interpolated_file_name.endswith('.parquet'):
            original_file_path = os.path.join(original_folder, interpolated_file_name)
            interpolated_file_path = os.path.join(interpolated_folder, interpolated_file_name)

            if os.path.exists(original_file_path):
                original_df = pd.read_parquet(original_file_path)
                interpolated_df = pd.read_parquet(interpolated_file_path)

                # Convert the index to a datetime index if it's not already
                original_df.index = pd.to_datetime(original_df.index)
                interpolated_df.index = pd.to_datetime(interpolated_df.index)

                # Filter by the specific date
                original_df = original_df[original_df.index.date == pd.to_datetime(specific_date).date()]
                interpolated_df = interpolated_df[interpolated_df.index.date == pd.to_datetime(specific_date).date()]

                # Create a figure with two subplots (vertical layout)
                plt.figure(figsize=(12, 7))

                # Plot for original data
                plt.subplot(2, 1, 1)  
                plt.plot(original_df.index, original_df[selected_column], linestyle='-', color='blue')
                plt.title(f'Original {selected_column} over Time')
                plt.xlabel('Time')
                plt.ylabel(selected_column)
                plt.xticks(rotation=45)
                plt.grid(True)

                # Plot for interpolated data
                plt.subplot(2, 1, 2)  
                plt.plot(interpolated_df.index, interpolated_df[selected_column], linestyle='-', color='red')
                plt.title(f'Interpolated {selected_column} over Time')
                plt.xlabel('Time')
                plt.ylabel(selected_column)
                plt.xticks(rotation=45)
                plt.grid(True)

                plt.tight_layout()  # Adjust layout to not overlap
                plt.show()