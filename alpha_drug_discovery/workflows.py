"""High level workflow utilities combining multiple modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional networkx dependency
    import networkx as nx
except ImportError as exc:  # pragma: no cover - executed when networkx missing
    nx = None  # type: ignore[assignment]
    _NETWORKX_IMPORT_ERROR = exc
else:
    _NETWORKX_IMPORT_ERROR = None

from data import dataset_download
try:  # pragma: no cover - biomarker module depends on optional sklearn
    from alpha_drug_discovery import biomarker_discovery
except ImportError as exc:  # pragma: no cover - executed when sklearn missing
    biomarker_discovery = None  # type: ignore[assignment]
    _BIOMARKER_IMPORT_ERROR = exc
else:
    _BIOMARKER_IMPORT_ERROR = None

try:  # pragma: no cover - optional sklearn dependency
    from alpha_drug_discovery import drug_repurposing
except ImportError as exc:  # pragma: no cover - executed when sklearn missing
    drug_repurposing = None  # type: ignore[assignment]
    _DRUG_REPURPOSING_IMPORT_ERROR = exc
else:
    _DRUG_REPURPOSING_IMPORT_ERROR = None

from alpha_drug_discovery import plugin_system

try:  # pragma: no cover - report generation depends on reportlab
    from alpha_drug_discovery import report_generation
except ImportError as exc:  # pragma: no cover - executed when reportlab missing
    report_generation = None  # type: ignore[assignment]
    _REPORT_IMPORT_ERROR = exc
else:
    _REPORT_IMPORT_ERROR = None
try:  # pragma: no cover - RDKit optional dependency
    from repurposing import automated_synthesis, network_drug_repurposing
except ImportError as exc:  # pragma: no cover - executed when RDKit missing
    automated_synthesis = None  # type: ignore[assignment]
    network_drug_repurposing = None  # type: ignore[assignment]
    _REPURPOSING_IMPORT_ERROR = exc
else:
    _REPURPOSING_IMPORT_ERROR = None

try:  # pragma: no cover - yaml optional dependency
    from utils import config as config_utils
except ImportError as exc:  # pragma: no cover - executed when PyYAML missing
    config_utils = None  # type: ignore[assignment]
    _CONFIG_IMPORT_ERROR = exc
else:
    _CONFIG_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency guard
    import torch  # noqa: F401  # Imported for type consistency in optional modules
except ImportError:  # pragma: no cover - executed when torch missing
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from alpha_drug_discovery import admet_prediction, generative_model, predictive_toxicology
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
except ImportError as exc:  # pragma: no cover - executed when torch missing
    _DL_IMPORT_ERROR = exc
    _TORCH_PIPELINE_AVAILABLE = False
    admet_prediction = generative_model = predictive_toxicology = None  # type: ignore[assignment]
    gan_drug_design = rl_drug_design = deep_docking = gnn_property_prediction = None  # type: ignore[assignment]
    ai_molecular_dynamics = qm_mm_simulation = protein_structure_prediction = integrative_biomarker_discovery = None  # type: ignore[assignment]
else:
    _DL_IMPORT_ERROR = None
    _TORCH_PIPELINE_AVAILABLE = True


def basic_drug_discovery_pipeline(config_path: str = "config.yaml") -> None:
    """Run a simple end-to-end pipeline using sample data.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML configuration file describing hyper-parameters.
    """
    if not _TORCH_PIPELINE_AVAILABLE:
        print("Deep learning dependencies are unavailable; skipping pipeline execution.")
        if _DL_IMPORT_ERROR is not None:
            print(f"Reason: {_DL_IMPORT_ERROR}")
        return
    cfg = {}
    if config_utils is not None:
        try:
            cfg = config_utils.load_config(config_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"Warning: failed to load config: {exc}")
    else:
        print("Skipping config loading due to missing dependency.")
        if _CONFIG_IMPORT_ERROR is not None:
            print(f"Reason: {_CONFIG_IMPORT_ERROR}")

    print("Loading dataset ...")
    df = dataset_download.load_dataset("esol", use_sample=True)
    X = df.select_dtypes(float).values

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
    if report_generation is not None:
        report_generation.create_report("pipeline_report.pdf", "Pipeline Summary", report_lines)
    else:
        print("Skipping report generation due to missing dependency.")
        if _REPORT_IMPORT_ERROR is not None:
            print(f"Reason: {_REPORT_IMPORT_ERROR}")
    print("Pipeline complete.")


def full_feature_pipeline(config_path: str = "config.yaml") -> None:
    """Demonstrate all major modules using synthetic data."""
    if not _TORCH_PIPELINE_AVAILABLE:
        print("Deep learning dependencies are unavailable; skipping full feature demo.")
        if _DL_IMPORT_ERROR is not None:
            print(f"Reason: {_DL_IMPORT_ERROR}")
        return
    cfg = {}
    if config_utils is not None:
        try:
            cfg = config_utils.load_config(config_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"Warning: failed to load config: {exc}")
    else:
        print("Skipping config loading due to missing dependency.")
        if _CONFIG_IMPORT_ERROR is not None:
            print(f"Reason: {_CONFIG_IMPORT_ERROR}")

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
    if biomarker_discovery is not None:
        biom_X = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
        biom_y = pd.Series(np.random.randint(0, 2, size=20))
        biomarkers = biomarker_discovery.discover_biomarkers(biom_X, biom_y, n_top_features=2)
    else:
        biomarkers = []
        print("Skipping biomarker discovery step due to missing dependency.")
        if _BIOMARKER_IMPORT_ERROR is not None:
            print(f"Reason: {_BIOMARKER_IMPORT_ERROR}")

    # 5) Drug repurposing networks
    if drug_repurposing is not None:
        drug_feats = np.random.rand(3, 4)
        target_feats = np.random.rand(5, 4)
        G = drug_repurposing.build_drug_target_network(drug_feats, target_feats)
        repurpose = drug_repurposing.identify_repurposing_opportunities(G, "drug_0")
        if network_drug_repurposing is not None and nx is not None:
            simple_graph = nx.path_graph(5)
            network_drug_repurposing.propagate_network(simple_graph, 0, steps=2)
        else:
            print("Skipping network propagation due to missing dependency.")
            reasons = []
            if _REPURPOSING_IMPORT_ERROR is not None:
                reasons.append(str(_REPURPOSING_IMPORT_ERROR))
            if nx is None and _NETWORKX_IMPORT_ERROR is not None:
                reasons.append(str(_NETWORKX_IMPORT_ERROR))
            if reasons:
                print("Reason: " + "; ".join(reasons))
    else:
        repurpose = []
        print("Skipping drug repurposing step due to missing dependency.")
        if _DRUG_REPURPOSING_IMPORT_ERROR is not None:
            print(f"Reason: {_DRUG_REPURPOSING_IMPORT_ERROR}")

    # 6) ADMET and toxicity prediction
    y_admet = np.random.rand(len(X), 5)
    admet_prediction.train_admet_model(X, y_admet, epochs=1)

    X_tox = np.random.rand(10, 6)
    y_tox = np.random.randint(0, 2, size=10)
    predictive_toxicology.train_toxicity_model(X_tox, y_tox, epochs=1)

    # 7) Automated synthesis example
    if automated_synthesis is not None:
        products = automated_synthesis.predict_reaction_outcome("CCO.O")
    else:
        products = None
        print("Skipping automated synthesis due to missing dependency.")
        if _REPURPOSING_IMPORT_ERROR is not None:
            print(f"Reason: {_REPURPOSING_IMPORT_ERROR}")

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
    if report_generation is not None:
        report_generation.create_report("full_demo_report.pdf", "Full Feature Pipeline", report_lines)
    else:
        print("Skipping report generation due to missing dependency.")
        if _REPORT_IMPORT_ERROR is not None:
            print(f"Reason: {_REPORT_IMPORT_ERROR}")
    print("Full feature pipeline complete.")
