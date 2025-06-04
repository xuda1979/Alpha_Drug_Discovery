# __init__.py

"""Model submodules for Alpha Drug Discovery.

This package avoids importing heavy dependencies (e.g. PyTorch) at import time.
Individual modules should be imported lazily by consumers when needed.
"""

__all__ = [
    'gan_drug_design',
    'rl_drug_design',
    'deep_docking',
    'gnn_property_prediction',
]
