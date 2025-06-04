"""High level workflow utilities combining multiple modules."""

from __future__ import annotations

import numpy as np

from data import dataset_download
from alpha_drug_discovery import (
    generative_model,
    admet_prediction,
    biomarker_discovery,
    drug_repurposing,
    predictive_toxicology,
    plugin_system,
    report_generation,
)
from models import (
    gan_drug_design,
    rl_drug_design,
    deep_docking,
    gnn_property_prediction,
    ai_molecular_dynamics,
    qm_mm_simulation,
    protein_structure_prediction,
    integrative_biomarker_discovery,
)
from repurposing import automated_synthesis, network_drug_repurposing
from utils import config as config_utils
import pandas as pd
import networkx as nx
import torch


def basic_drug_discovery_pipeline(config_path: str = "config.yaml") -> None:
    """Run a simple end-to-end pipeline using sample data.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML configuration file describing hyper-parameters.
    """
    cfg = {}
    try:
        cfg = config_utils.load_config(config_path)
    except FileNotFoundError:
        pass

    print("Loading dataset ...")
    df = dataset_download.load_dataset("esol", use_sample=False, cache_dir="data")
    X = df.select_dtypes(include=[np.number]).values # Ensure we select all numeric types

    if X.shape[0] > 0 and X.shape[1] > 0: # Ensure X is not empty and has features
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        # Add epsilon to prevent division by zero if a column has all same values
        X = (X - x_min) / (x_max - x_min + 1e-8)
        X = np.nan_to_num(X, nan=0.0) # Handle potential NaNs if x_max - x_min was 0 for a feature
    else:
        # Handle case where X might be empty after select_dtypes or if esol.csv was empty/no numeric
        print("Warning: ESOL dataset has no numeric data or is empty. VAE training will use random data.")
        X = np.random.rand(100, 10).astype(np.float32) # Fallback to random data, already in [0,1]

    latent_dim = cfg.get("latent_dim", 8)
    epochs = cfg.get("vae_epochs", 2)
    vae_model = generative_model.train_vae(X, latent_dim=latent_dim, epochs=epochs)

    print("Generating molecules ...")
    molecules = generative_model.generate_new_molecules(vae_model, num_samples=5)

    print("Training ADMET model ...")
    y = np.random.rand(len(X), 5)
    admet_model = admet_prediction.train_admet_model(X, y, epochs=2)

    print("Creating report ...")
    report_lines = [
        "Generated molecules: " + str(molecules.shape),
        "VAE latent_dim: " + str(latent_dim),
    ]
    report_generation.create_report("pipeline_report.pdf", "Pipeline Summary", report_lines)
    print("Pipeline complete.")


def full_feature_pipeline(config_path: str = "config.yaml") -> None:
    """Demonstrate all major modules using synthetic data."""
    cfg = {}
    try:
        cfg = config_utils.load_config(config_path)
    except FileNotFoundError:
        pass

    # Load sample dataset
    df = dataset_download.load_dataset("esol", use_sample=True)
    X = df.select_dtypes(float).values

    # 1) Train generative models (VAE and GAN)
    vae_model = generative_model.train_vae(X, latent_dim=cfg.get("latent_dim", 8), epochs=1)
    gan_drug_design.train_gan(X, epochs=1, batch_size=min(32, len(X)))

    # 2) Apply reinforcement learning with a dummy environment
    class DummyEnv:
        def __init__(self, state_dim: int, action_dim: int, max_steps: int = 5):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.max_steps = max_steps

        def reset(self):
            return np.zeros(self.state_dim)

        def step(self, action: int):
            next_state = np.random.rand(self.state_dim)
            reward = np.random.rand()
            done = False
            return next_state, reward, done, {}

    env = DummyEnv(10, 3)
    policy = rl_drug_design.PolicyNetwork(state_dim=10, action_dim=3)
    rl_drug_design.train_policy_gradient(env, policy, epochs=1)

    # 3) Deep docking with random grids
    X_dock = np.random.rand(8, 1, 8, 8, 8).astype(np.float32)
    y_dock = np.random.rand(8).astype(np.float32)
    deep_docking.train_docking_model(X_dock, y_dock, epochs=1)

    # 4) Biomarker discovery
    biom_X = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
    biom_y = pd.Series(np.random.randint(0, 2, size=20))
    biomarkers = biomarker_discovery.discover_biomarkers(biom_X, biom_y, n_top_features=2)

    # 5) Drug repurposing networks
    drug_feats = np.random.rand(3, 4)
    target_feats = np.random.rand(5, 4)
    G = drug_repurposing.build_drug_target_network(drug_feats, target_feats)
    repurpose = drug_repurposing.identify_repurposing_opportunities(G, "drug_0")

    simple_graph = nx.path_graph(5)
    network_drug_repurposing.propagate_network(simple_graph, 0, steps=2)

    # 6) ADMET and toxicity prediction
    y_admet = np.random.rand(len(X), 5)
    admet_prediction.train_admet_model(X, y_admet, epochs=1)

    X_tox = np.random.rand(10, 6)
    y_tox = np.random.randint(0, 2, size=10)
    predictive_toxicology.train_toxicity_model(X_tox, y_tox, epochs=1)

    # 7) Automated synthesis example
    products = automated_synthesis.predict_reaction_outcome("CCO.O")

    # 8) Plugin execution
    for plugin in plugin_system.discover_plugins():
        plugin_system.run_plugin(plugin, {"message": "Running plugin"})

    # 9) Graph neural network property prediction
    g_features = [np.random.rand(4, 3) for _ in range(3)]
    g_adjs = [np.eye(4) for _ in range(3)]
    g_labels = np.random.randint(0, 2, size=3)
    gnn_property_prediction.train_gcn(g_features, g_adjs, g_labels, epochs=1)

    # 10) Protein structure prediction
    X_prot = np.random.rand(10, 20)
    y_prot = np.zeros((10, 3))
    y_prot[np.arange(10), np.random.randint(0, 3, size=10)] = 1
    protein_structure_prediction.train_protein_model(X_prot, y_prot, epochs=1)

    # 11) Integrative biomarker model
    X_genomics = np.random.rand(10, 5)
    X_proteomics = np.random.rand(10, 5)
    X_metabolomics = np.random.rand(10, 5)
    y_multi = np.random.randint(0, 2, size=10)
    integrative_biomarker_discovery.train_integrative_biomarker_model(
        X_genomics, X_proteomics, X_metabolomics, y_multi, epochs=1
    )

    # 12) AI molecular dynamics and QM/MM
    X_dyn = np.random.rand(10, 3)
    ai_molecular_dynamics.train_molecular_dynamics_model(X_dyn, X_dyn, epochs=1)

    qm_mm_simulation.qm_mm_simulation(torch.randn(1, 3))

    # Generate summary report
    report_lines = [
        f"Biomarkers: {biomarkers}",
        f"Repurposing: {repurpose}",
        f"Synth products: {products}",
    ]
    report_generation.create_report("full_demo_report.pdf", "Full Feature Pipeline", report_lines)
    print("Full feature pipeline complete.")
