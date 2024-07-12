import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import logging 

def nearest_cluster(locations, cluster_size, min_max):
    coordinates = locations[['Latitude', 'Longitude']]
    distance_matrix = squareform(pdist(coordinates, metric='euclidean'))
    np.fill_diagonal(distance_matrix, np.inf)

    clusters = []
    unassigned = set(range(len(coordinates)))

    while len(unassigned) >= cluster_size:
        min_dist_idx = min(unassigned, key=lambda x: np.min(distance_matrix[x]))
        closest_indices = np.argsort(distance_matrix[min_dist_idx])[:cluster_size-1]
        cluster = [min_dist_idx] + closest_indices.tolist()
        clusters.append(cluster)
        unassigned -= set(cluster)
        for idx in cluster:
            distance_matrix[:, idx] = np.inf
            distance_matrix[idx, :] = np.inf

    clustered_farms = {f'Cluster{i+1}': locations.iloc[indices][locations.columns[0]].tolist() for i, indices in enumerate(clusters)}
    sorted_farms = [farm for cluster in clustered_farms.values() for farm in cluster]
    return sorted_farms

def random_cluster(locations, cluster_size, min_max):
    shuffled_indices = np.random.permutation(len(locations))
    
    clusters = [shuffled_indices[i:i + cluster_size] for i in range(0, len(shuffled_indices), cluster_size)]
    
    if isinstance(locations, pd.DataFrame):
        clustered_farms = {f'Cluster{i+1}': locations.iloc[indices][locations.columns[0]].tolist() for i, indices in enumerate(clusters)}
    else:  
        clustered_farms = {f'Cluster{i+1}': [locations[i] for i in indices] for i, indices in enumerate(clusters)}
    
    sorted_farms = [farm for cluster in clustered_farms.values() for farm in cluster]
    return sorted_farms

def capacity_cluster(locations, cluster_size, max_values):
    farm_names = locations['Abbreviation'].tolist()
    farm_max_values = []
    
    for farm in farm_names:
        if farm in max_values.columns:
            farm_max_value = max_values[farm].max()
            farm_max_values.append((farm, farm_max_value))
        else:
            logging.info(f"Warning: Farm {farm} not found in max_values DataFrame.")
    farm_max_values.sort(key=lambda x: x[1], reverse=True)
    sorted_farms = [farm for farm, _ in farm_max_values]
    
    return sorted_farms