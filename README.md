# Attention-Based Recurrent Neural-Network for Joint Probabilistic Wind Power Forecasting of Spatial Correlated Wind Farms

<img src="Images/windturbine.gif" alt="Wind Turbine" style="width: 100%;">


*Image Source: "Breitenlee-VESTAS-V-52 wind turbine" by [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Breitenlee-VESTAS-V-52_wind_turbine_looped.gif)*


## 🤖 OVERVIEW 🦾
This project centers around the employment of an Attention-Based Recurrent Neural Network, for joint probabilistic forecasting of wind power of spatially correlated wind farms. 

## 🧑‍🔬 FEATURES 🧑‍🔧️
- **LSTM-based Models**: Utilizes LSTM networks with attention mechanisms to adeptly capture temporal sequences and dependencies across multiple time series.

- **Data Preprocessing**: Streamlined preprocessing pipeline for data cleansing, normalization, and preparation, ensuring high-quality inputs for modeling.

- **Flexible Configuration**: Easily adjustable model parameters and experiment settings through JSON configurations, fostering extensive experimentation and tuning.

- **Training and Evaluation**: Automated scripts for training models with saved parameters and evaluating performance using standard metrics (MAE, MSE, RMSE).

- **Visualization Utilities**: Built-in tools for plotting predictions and analyzing model performance, supporting libraries like Matplotlib and Seaborn for insightful visualizations.

- **Hyperparameter Optimization**: Integrated hyperparameter tuning to identify optimal model configurations, improving accuracy and efficiency.


## 👮🏼‍♂ REQUIREMENTS 📢
- matplotlib==3.8.2
- numpy==1.26.2
- pandas==2.1.3
- torch==2.1.1
- alive-progress==3.1.5
- scipy==1.11.4
- scikit-learn==1.3.2
- plotly==5.18.0
- seaborn==0.13.0
- bokeh==3.3.4
- optuna==3.5.0

## 📂 DIRECTORY LAYOUT 📐
```plaintext
simon-faltz/
│
├── data/      
│   │   ├── weather                   # Cached Weather Dataset
│   │   ├── wind farm data            # Cached Generation Dataset
│   │   └── spatial                   # Cached Spatial Dataset              
│   ├── cache
│   ├── wind_farm_data/               
│   │   ├── excel/                    # Wind Farm Generation Output data                              
│   │   └── parquet/                  # Generation Data as Parquet Files
│   └── weather_data/      
│       ├── weather/                  # Raw weather data files.
│       └── weather_interpolated/     # Interpolated weather data files.
│
├── parameters/                       # Trained model parameters.
│
├── results/                          
│   ├── plots/                        # Plots Generated from Testing
│   └── spatial/                      # Visualised Spatial Data 
│
├── scripts/  
│   ├── config_manager.py             # Helping Script for Config Loading.
│   ├── train.py                      # Script for Training the Model.
│   └── test.py                       # Script for Testing the Model.
│
├── src/                              
│   ├── models/                       
│   │   ├── encoder.py                # Encoder Part of the LSTM Model.
│   │   ├── decoder.py                # Decoder Part of the LSTM Model.
│   │   └── attention.py              # Attention Mechanisms.
│   ├── preprocessing/                # Data Preprocessing Utilities.
│   │   ├── dataset.py                # Dataset preparation and loading utilities.
│   │   ├── wind_data.py              # Generation data preprocessing.
│   │   ├── spatial_data.py           # Spatial data preprocessing.
│   │   ├── utils.py                  # Helping Scripts for data processing.
│   │   └── weather_data.py           # Weather data preprocessing.
│   ├── evaluation/                   # Model evaluation metrics and visualization.
│   │   ├── metrics.py                # Evaluation metrics.
│   │   └── plotter.py                # Plotting utilities.
│   └── utils/                        
│       ├── file_operations.py        # File read/write utilities.
│       └── transformations.py        # Data transformation utilities.
│
│
├── config.jsno                       # Config File for Parameter modification.
├── requirements.txt                  # Required Python packages and dependencies.
└── README.md                         # Comprehensive project documentation and instructions.
```

## 👟 USAGE 🎡

For usage, type the following line into the terminal. Deside weather train oder test, and which config file to use. Inside the config file, you can manage several parameters.
```
python main.py --mode [train/test] --config path/to/config.json
```

## 🏋🏻 TRAIN 🚂
Finally, to train the model, the command could look like:

```
python main.py --mode train --config config.json

```

## 🧪 TEST ⚗️
Accordinly, to test the model, the command could look like:
```
python main.py --mode train --config config.json
```

## 🏁 HYPERPARAMETERTUNING 🏁
This project includes a systematic approach to explore a wide range of configurations to identify the most effective settings. To run the hyperparametertuning, type:

```
python racereadytuning.py
```

inside `/src/hyperparameter_tuning/racereadytuning.py` you can modify the wished parameters. The code will iterate threw every combination. E.g.: 
```
hyperparameters = {
    'sequence_length': [24, 48, 72, 108, 144],
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 3],
    'epochs': [200],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1]
}
```

## ⚙️ CONFIG FILE 🦿

Inside the config file, you can modify several different parameters for different modes. They are shown below:

### 🎛️ Common Settings 🎛️

Common settings are applied across both the training and testing phases.

- `data_folder`: Path to the directory containing your dataset.
- `sequence_length`: Number of time steps for each input time series sequence.
- `prediction_length`: Number of time steps the model should predict into the future.
- `num_series`: Number of distinct time series to be fed into the model.
- `hidden_size`: Size of the hidden layers within the neural network model.
- `num_layers`: Number of layers in the neural network model.
- `num_directions`: 1 for unidirectional, 2 for bidirectional LSTM.
- `normalization`: Method used to normalize the data.
- `cluster_type`: Strategy used for clustering time series data.

### 🏃 Training Settings 🏋️

Settings specific to the model training phase.

- `epochs`: Total number of training cycles through the dataset.
- `learning_rate`: Step size at each iteration of the model optimization.
- `data_path`: Relative path to the training data file.
- `loss_function`: Loss function utilized for training.

### 📝 Testing Settings 📒

Configuration for the testing phase.

- `plot`: Enables plotting of model predictions if set to true.
- `data_path`: Path to the testing data.
- `plot_type`: Library to use for plotting (e.g., matplotlib, seaborn).
- `parameters_path`: Location of the saved model parameters for testing.

### ⛈️ Weather Features 🌬️

Determines which weather-related features are included in the data.

- Each key represents a specific weather feature, where `true` includes the feature, and `false` excludes it from the model.

### 🌐 Spatial 🌏

- `plot`: Controls whether spatial analysis plots are generated.

## 🔎 GRAFICAL ABSTRACT 🎨

<img src="Images/graficalabstract.png" alt="Grafical Abstract" style="width: 100%;">