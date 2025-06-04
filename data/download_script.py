import os
import pandas as pd
from dataset_download import download_esol_dataset, download_chembl_activity_data, download_bindingdb_dataset

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

print("Downloading ESOL dataset...")
esol_df = download_esol_dataset(path="data/esol.csv")
print(f"ESOL dataset downloaded to data/esol.csv, shape: {esol_df.shape}")

print("\nDownloading ChEMBL activity data for Aspirin (CHEMBL25)...")
# For ChEMBL, the function returns a DataFrame which we then save.
# The original download_chembl_activity_data doesn't save to file itself.
chembl_df = download_chembl_activity_data(molecule_chembl_id="CHEMBL25", limit=100)
chembl_output_path = "data/chembl_aspirin_activity.csv"
chembl_df.to_csv(chembl_output_path, index=False)
print(f"ChEMBL data for CHEMBL25 downloaded to {chembl_output_path}, shape: {chembl_df.shape}")

print("\nDownloading BindingDB dataset...")
# The download_bindingdb_dataset function handles saving the .tsv file itself and returns a DataFrame.
bindingdb_df = download_bindingdb_dataset(path="data/bindingdb_all.tsv")
print(f"BindingDB dataset downloaded to data/bindingdb_all.tsv, shape: {bindingdb_df.shape}")

print("\nAll dataset downloads attempted.")
