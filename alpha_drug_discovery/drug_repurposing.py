# drug_repurposing.py

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_drug_target_network(drug_features, target_features):
    """
    Build a drug-target interaction network based on cosine similarity.
    
    Parameters:
    drug_features (np.ndarray): Feature matrix of drugs.
    target_features (np.ndarray): Feature matrix of targets.

    Returns:
    nx.Graph: Drug-target interaction network.
    """
    similarity_matrix = cosine_similarity(drug_features, target_features)
    G = nx.Graph()
    
    for i, drug in enumerate(drug_features):
        for j, target in enumerate(target_features):
            similarity_score = similarity_matrix[i, j]
            if similarity_score > 0.5:  # threshold to consider an interaction
                G.add_edge(f"drug_{i}", f"target_{j}", weight=similarity_score)
    
    return G

def identify_repurposing_opportunities(G, drug_of_interest):
    """
    Identify potential repurposing opportunities by analyzing the network.
    
    Parameters:
    G (nx.Graph): Drug-target interaction network.
    drug_of_interest (str): Identifier of the drug to be repurposed.
    
    Returns:
    list: List of potential target interactions.
    """
    neighbors = list(nx.neighbors(G, drug_of_interest))
    return neighbors
