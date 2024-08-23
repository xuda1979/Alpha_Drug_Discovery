# data/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame by handling missing values, duplicates, etc.
    
    Parameters:
    df (pd.DataFrame): The input data frame.
    
    Returns:
    pd.DataFrame: The cleaned data frame.
    """
    df = df.drop_duplicates()
    df = df.dropna()  # or df.fillna(method='ffill') depending on use case
    return df

def normalize_data(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Normalize the input DataFrame using the specified method.
    
    Parameters:
    df (pd.DataFrame): The input data frame.
    method (str): The normalization method ('standard' or 'minmax').
    
    Returns:
    pd.DataFrame: The normalized data frame.
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a series of preprocessing steps to the input DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input data frame.
    
    Returns:
    pd.DataFrame: The preprocessed data frame.
    """
    df = clean_data(df)
    df = normalize_data(df)
    return df
