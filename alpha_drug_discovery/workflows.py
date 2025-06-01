"""High level workflow utilities combining multiple modules."""

from __future__ import annotations

import numpy as np

from data import dataset_download
from alpha_drug_discovery import generative_model, admet_prediction, report_generation
from utils import config as config_utils


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
    report_generation.create_report("pipeline_report.pdf", "Pipeline Summary", report_lines)
    print("Pipeline complete.")
