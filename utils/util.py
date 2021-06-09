import os
from typing import Tuple
import yaml
import sys
import numpy as np
import pandas as pd
import pickle

from scipy.io import loadmat
from scipy.spatial import distance

# Import all models that can be used by str_to_class(str)
from models.conv_ae2 import ConvAE2 # Used NN for federated learning (Autoencoder)
from models.conv_ae3 import ConvAE3 # Used NN for federated learning (Autoencoder)
from utils.dataset import LibriData

# Import configuration file
def import_config(config_file: str) -> dict:
    """Import config file to dict.

    Args:
        config_file (str): Path to config.yml file.

    Returns:
        dict: Dict containing the confuguration
    """
    with open(config_file) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    return config

# Select class from string provided by config
def str_to_class(str: str) -> type:
    """Gets class type from string.

    Args:
        str (str): Name of class.

    Returns:
        type: Type of class for initializing object.
    """
    return getattr(sys.modules[__name__], str)

# Calculates the critical distance
def critical_distance(V: float, t60: float, c: float=341) -> float:
    """Computes critical distance based on Sabine's equation.

    Args:
        V (float): Room volume.
        t60 (float): Broadband reverberation time T60 of room.
        c (float, optional): Speed of sound. Defaults to 341.

    Returns:
        float: Critical distance in meter.
    """

    r_H = np.sqrt(V / (c * t60))
    return r_H

# Sort by diagonal
from typing import Union
def mean_d_matrix(cts_d: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Sorts CTS matrix in the diagonal

    Used to sort the CTS matrix so the closest cluster to source 1 is always in the beginning and to source two the next.

    Args:
        cts_d (Union[np.ndarray, pd.DataFrame]): Normalized cluster-to-source distance

    Returns:
        np.ndarray: Sorted normalized cluster-to-source distance
    """
    if isinstance(cts_d, pd.DataFrame):
        cts_d = cts_d.to_numpy()

    src = 0
    sorted = []
    idc = [*range(0, len(cts_d[0]))]
    while len(idc) > 0:
        sorted.append(idc[np.argsort(cts_d[src][idc], axis=0)[0]])
        idc.remove(sorted[-1])
        src = src + 1 if src+1 < len(cts_d) else 0


    return cts_d[:,sorted]

# Calculation of the cluster-to-source distance
def normalized_cluster_to_source_distance(clusters: list, clients: list, memberships:list , config: dict) -> pd.DataFrame:
    """Computation of the normalized cluster-to-source distance

    Args:
        clusters (list of list): Cluster configuration
        clients (list of Client): Client objects containing the node positions
        memberships (list): Cluster membership value for every client
        config (dict): Configuration

    Returns:
        pd.DataFrame: Normalized cluster-to-source distance matrix
    """
    from scipy.spatial import distance
    import itertools

    positions = []
    matlab_data = loadmat(config['paths']['ir-meta-dir'])

    # Clusters
    clusters = [x.tolist() if isinstance(x, np.ndarray) else x for x in clusters]
    clusters = [x if isinstance(x, list) else [x] for x in clusters]
    clusters = [x for x in clusters if x != []]

    # Compute centroids
    centroids = []
    for cluster in clusters:
        if memberships is not None:
            weights = [memberships[client_idx] for client_idx in cluster]
        else:
            weights = None

        positions_cluster = []
        for node in cluster:
            try:
                positions_cluster.append(clients[node].data.position)
            except AttributeError as e:
                positions_cluster.append(clients[node].position)
        centroids.append(np.average(positions_cluster, axis=0, weights=weights))
    positions.extend(centroids)
    
    # Get source positions
    sources = []
    for i in range(matlab_data["nsrc"][0][0]):
        sources.append([matlab_data["src_positions"][0][i], matlab_data["src_positions"][1][i], matlab_data["src_positions"][2][i]])
    positions.extend(sources)
    
    # Get mean distance between all sources for normalization
    sources_dist = []
    for pair in itertools.combinations(sources, 2):
        sources_dist.append(distance.euclidean(pair[0], pair[1]))
    norm_src_d = np.mean(sources_dist)

    # Normalized Cluster-to-Source distance (Gergen: P. 78)
    d = np.zeros((len(sources), len(centroids)))
    for q, r_s in enumerate(sources):
        for n, r_n in enumerate(centroids):
            d[q][n] = distance.euclidean(r_s, r_n) / norm_src_d

    # Sort CTS in diagonal
    d = mean_d_matrix(d)
    # Create DataFrame
    df = pd.DataFrame(d)
    df.columns = [r"$C_%d$" % (i) for i in range(len(centroids))]
    df.index = ["%s" % (chr(matlab_data["src_ids"][0][i]) + chr(matlab_data["src_ids"][1][i])) for i in range(len(sources))]
    
    return df

def load_object(path: str) -> dict:
    """Load pickle file.

    If loading a speaker_split file, the result is a dictionary containing gender and id of the speakers from this file.

    Args:
        path (str): Path to pickle file

    Returns:
        dict: Data saved in the pickle file.
    """
    with open(path, 'rb') as pkl_input:
        return pickle.load(pkl_input)

# List all directories in folder
def listdirs(folder: str) -> list:
    """Lists all directory in folder path.

    Args:
        folder (str): Folder path.

    Returns:
        list: List of directories in folder.
    """
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

# Calculate membership values
def get_memberships(clusters_t: list, similarities: np.ndarray, weight: float=0.5) -> Tuple[list, np.ndarray, np.ndarray]:
    """Computes cluster membership values.

    Args:
        clusters_t (list of lists/ndarrays): Cluster configuration containing client ids.
        similarities (np.ndarray): Cosine similarity matrix.
        weight (float, optional): Lambda factor for finding reference node. (cross 0 - 1 intra) Defaults to 0.5.

    Returns:
        Tuple[list, np.ndarray, np.ndarray]: Normalized membership values, intra cluster cosine similarities, cross cluster cosine similarities.
    """
    # Change format of cluster list
    clusters = [x.tolist() if isinstance(x, np.ndarray) else x for x in clusters_t]
    clusters = [x if isinstance(x, list) else [x] for x in clusters]
    clusters = [x for x in clusters if x != []]
    # Get number of clients
    n_clients = len(sum(clusters, []))
    # Init values
    intra_cluster = np.zeros(n_clients)
    cross_cluster = np.zeros(n_clients)
    norm_memberships = np.zeros(n_clients)
    # Compute for each cluster
    for cluster_idx, cluster in enumerate(clusters):
        # For each client in the current cluster
        for client in cluster:
            # All clients except current
            tmp_cluster = cluster.copy()
            tmp_cluster.remove(client) 
            # Compute mean intra cluster cosine similarity for current client
            intra_cluster[client] = np.mean([similarities[client][idx] for idx in tmp_cluster])
            if (cluster_idx+1) < len(clusters):
                tmp_cluster = clusters[cluster_idx+1].copy()
            else:
                tmp_cluster = clusters[cluster_idx-1].copy()
            # Compute mean cross cluster cosine similarity for current client
            cross_cluster[client] = np.mean([similarities[client][idx] for idx in tmp_cluster])

    # Weight Range: (cross) 0 - 1 (intra)
    # Normalize intra and cross cluster similarities
    amin, amax = min(intra_cluster), max(intra_cluster)
    norm_intra = np.multiply([(val-amin) / (amax-amin) for val in intra_cluster], weight)
    amin, amax = min(cross_cluster), max(cross_cluster)
    norm_cross = np.multiply([(val-amin) / (amax-amin) for val in cross_cluster], 1-weight)

    # Get min MV in cluster as reference node
    # Compute similarity to reference
    memberships = np.zeros(n_clients)
    for cluster in clusters:
        # Get min MV Value
        min_idx = cluster[0]
        min_mv = norm_intra[cluster[0]] + norm_cross[cluster[0]]
        for client in cluster:
            if norm_intra[client] + norm_cross[client] < min_mv:
                min_idx = client
                min_mv = norm_intra[client] + norm_cross[client]
        
        # Similarity to reference node
        for client in cluster:
            memberships[client] = similarities[client][min_idx]

    # Normalize cluster membership values
    amin, amax = min(memberships), max(memberships)
    norm_memberships = [(val-amin) / (amax-amin) for val in memberships]
    
    if weight != 0:
        return norm_memberships, np.multiply(norm_intra, 1/weight), np.multiply(norm_cross, 1/(1-weight))
    else:
        return norm_memberships, np.multiply(norm_intra, weight), np.multiply(norm_cross, 1/(1-weight))