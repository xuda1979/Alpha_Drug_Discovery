# network_drug_repurposing.py

import networkx as nx
import numpy as np

def propagate_network(G, drug_of_interest, steps=3):
    """
    Propagate information through the network to find repurposing opportunities.
    
    Parameters:
    G (nx.Graph): Drug-target interaction network.
    drug_of_interest (str): The drug to analyze.
    steps (int): Number of propagation steps.
    
    Returns:
    dict: Nodes with highest propagated scores.
    """
    # Initialize scores
    scores = {node: 0 for node in G.nodes()}
    scores[drug_of_interest] = 1

    # Propagation
    for _ in range(steps):
        new_scores = scores.copy()
        for node in G.nodes():
            for neighbor in G.neighbors(node):
                new_scores[neighbor] += scores[node] * G[node][neighbor].get('weight', 1)
        scores = new_scores

    # Sort nodes by propagated scores
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

def identify_repurposing_candidates(G, drug_of_interest, threshold=0.5):
    """
    Identify potential repurposing candidates based on network propagation.
    
    Parameters:
    G (nx.Graph): Drug-target interaction network.
    drug_of_interest (str): The drug to analyze.
    threshold (float): Score threshold to consider a candidate.
    
    Returns:
    list: List of potential targets or drugs for repurposing.
    """
    propagated_scores = propagate_network(G, drug_of_interest)
    candidates = [node for node, score in propagated_scores if score >= threshold]
    return candidates
