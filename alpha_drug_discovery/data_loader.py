import pandas as pd

def load_genomics_data(file_path):
    """
    Load genomics data from a CSV file.

    Parameters:
    file_path (str): Path to the genomics data CSV file.

    Returns:
    pd.DataFrame: Loaded genomics data.

    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the file is not a valid CSV.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.ParserError:
        raise ValueError(f"File could not be parsed as CSV: {file_path}")
    return data

def load_clinical_data(file_path):
    """
    Load clinical data from a CSV file.

    Parameters:
    file_path (str): Path to the clinical data CSV file.

    Returns:
    pd.DataFrame: Loaded clinical data.

    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the file is not a valid CSV.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.ParserError:
        raise ValueError(f"File could not be parsed as CSV: {file_path}")
    return data
