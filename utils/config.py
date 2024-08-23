# utils/config.py

import yaml

def load_config(config_path='config.yaml'):
    """
    Load a configuration from a YAML file.
    
    Parameters:
    config_path (str): Path to the YAML configuration file.
    
    Returns:
    dict: The configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(config, config_path='config.yaml'):
    """
    Save a configuration dictionary to a YAML file.
    
    Parameters:
    config (dict): The configuration dictionary to save.
    config_path (str): Path to the YAML configuration file.
    """
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
