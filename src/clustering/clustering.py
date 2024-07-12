import pandas as pd
from src.clustering.strategys import nearest_cluster, capacity_cluster, random_cluster

def cluster(farms, cluster_size, cluster_type, min_max):    
    locations = read_farm_locations(farms)
    clustering_strategy = cluster_farm_strategy(cluster_type)
    sorted_farms = clustering_strategy(locations, cluster_size, min_max)
    return sorted_farms

def read_farm_locations(farms):
    locations = pd.read_excel('data/wind_farm_data/locations_wind_farms.xlsx')
    return locations[locations['Abbreviation'].isin(farms)]

def cluster_farm_strategy(cluster_type):
    if cluster_type == 'nearest':
        return nearest_cluster
    if cluster_type == 'capacity':
        return capacity_cluster
    if cluster_type == 'random':
        return random_cluster