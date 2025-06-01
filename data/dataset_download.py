import os
import pandas as pd
import requests

ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
BINDINGDB_URL = "https://www.bindingdb.org/bind/resourcedownloads/BindingDB_All.tsv.zip"


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


def download_chembl_activity_data(molecule_chembl_id, limit=100):
    """Download activity data for a molecule from the ChEMBL API.

    Parameters
    ----------
    molecule_chembl_id : str
        ChEMBL identifier of the molecule (e.g. ``"CHEMBL25"``).
    limit : int, optional
        Maximum number of records to fetch.

    Returns
    -------
    pd.DataFrame
        Activity records returned by the API.
    """
    params = {"molecule_chembl_id": molecule_chembl_id, "limit": limit, "format": "json"}
    response = requests.get(CHEMBL_ACTIVITY_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    activities = data.get("activities", [])
    return pd.DataFrame(activities)


def download_bindingdb_dataset(path="data/bindingdb_all.tsv", use_sample=False):
    """Download the BindingDB dataset.

    Parameters
    ----------
    path : str
        Destination path of the TSV file.
    use_sample : bool, optional
        If ``True`` only the first 1000 rows are returned after download.

    Returns
    -------
    pd.DataFrame
        BindingDB dataset as a DataFrame.
    """
    import zipfile

    zip_path = path + ".zip"
    if not os.path.exists(path):
        if not os.path.exists(zip_path):
            response = requests.get(BINDINGDB_URL, timeout=10)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(response.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(path))

    df = pd.read_csv(path, sep="\t")
    if use_sample:
        df = df.head(1000)
    return df
