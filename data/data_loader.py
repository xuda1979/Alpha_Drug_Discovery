# data_loader.py

import pandas as pd

def load_csv_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)

def load_excel_data(file_path, sheet_name=0):
    """
    Load data from an Excel file into a Pandas DataFrame.
    
    Parameters:
    file_path (str): Path to the Excel file.
    sheet_name (str or int): Name or index of the sheet to load.
    
    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)

def load_json_data(file_path):
    """
    Load data from a JSON file into a Pandas DataFrame.
    
    Parameters:
    file_path (str): Path to the JSON file.
    
    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_json(file_path)

def load_tsv_data(file_path):
    """
    Load data from a TSV (Tab-Separated Values) file into a Pandas DataFrame.
    
    Parameters:
    file_path (str): Path to the TSV file.
    
    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path, sep='\t')

def load_data(file_path, file_type='csv', **kwargs):
    """
    Load data from a specified file type into a Pandas DataFrame.
    
    Parameters:
    file_path (str): Path to the file.
    file_type (str): Type of the file ('csv', 'excel', 'json', 'tsv').
    
    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    loaders = {
        'csv': load_csv_data,
        'excel': load_excel_data,
        'json': load_json_data,
        'tsv': load_tsv_data
    }
    
    if file_type in loaders:
        return loaders[file_type](file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
