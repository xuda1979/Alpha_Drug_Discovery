import os
import pandas as pd
import requests

ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"


def download_esol_dataset(path="data/esol.csv", use_sample=False):
    """Download the ESOL solubility dataset if not present and return a DataFrame.

    Parameters
    ----------
    path : str
        Local path to save the dataset.
    use_sample : bool
        If True, load the small sample dataset included with the repository.

    Returns
    -------
    pd.DataFrame
        Dataset as a DataFrame.
    """
    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "esol_sample.csv")
        return pd.read_csv(sample_path)

    if not os.path.exists(path):
        response = requests.get(ESOL_URL, timeout=10)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
    return pd.read_csv(path)
