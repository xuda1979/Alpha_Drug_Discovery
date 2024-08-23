# utils/helpers.py

import os
import torch

def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Parameters:
    directory (str): Path of the directory to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, path):
    """
    Save a PyTorch model to a specified path.
    
    Parameters:
    model (torch.nn.Module): The PyTorch model to save.
    path (str): The path where the model should be saved.
    """
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

def load_model(model_class, path):
    """
    Load a PyTorch model from a specified path.
    
    Parameters:
    model_class (torch.nn.Module): The class of the model to load.
    path (str): The path from which to load the model.
    
    Returns:
    torch.nn.Module: The loaded PyTorch model.
    """
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model
