# Alpha Drug Discovery

**Alpha Drug Discovery** is an experimental toolkit demonstrating machine learning techniques for modern drug discovery.  The project collects small examples such as GAN based molecule generation, reinforcement learning agents, network propagation for drug repurposing and more.  Every module is self contained and can be used independently for educational purposes.

## Features

- **GAN-based drug design** – generate novel compounds with a generative adversarial network.
- **Reinforcement learning** – optimise candidate molecules using policy gradient methods.
- **Deep docking** – simple 3D convolutional network for predicting docking scores.
- **Biomarker discovery** – identify important features from omics data with random forests.
- **Drug repurposing** – network propagation and reaction prediction utilities.
- **Graph neural networks** – property prediction with a small GCN example.
- **Basic visualisation utilities** for quick data exploration.
- **ADMET prediction** – estimate absorption, distribution, metabolism, excretion
  and toxicity with a simple neural network.
- **Plugin framework** – drop new algorithms into the `plugins/` folder.
- **Command line interface** – run predefined tasks via `run.py`.
- **End-to-end workflow** – sample pipeline combining generation and ADMET.
- **Docker support** – build a container for easy deployment.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Some modules require additional scientific packages (e.g. PyTorch, RDKit).  If a dependency is missing the corresponding feature can be skipped.

## Usage

Each module can be executed individually.  Example usage for training the GAN model:

```python
from models.gan_drug_design import train_gan
import torch

X = torch.randn(100, 10)
model = train_gan(X, epochs=10)
```

Training the new graph neural network model requires node features and adjacency matrices for each molecule:

```python
from models.gnn_property_prediction import train_gcn
import numpy as np

# Example synthetic dataset with 3 molecules
features = [np.random.rand(4, 3) for _ in range(3)]
adjs = [np.eye(4) for _ in range(3)]
labels = np.random.randint(0, 2, size=3)
gnn_model = train_gcn(features, adjs, labels, epochs=2)
```

Predicting ADMET properties for a set of molecules is similarly straightforward:

```python
from alpha_drug_discovery.admet_prediction import train_admet_model
import numpy as np

X = np.random.rand(10, 8)
y = np.random.rand(10, 5)
admet_model = train_admet_model(X, y, epochs=2)
```

For a step-by-step walkthrough including saving models and adjusting hyperparameters, consult
[the detailed ADMET prediction guide](docs/admet_prediction_tutorial.md).

See the `run.py` script for a minimal command line entry point combining several components.

For an automated demonstration run:

```bash
python run.py pipeline
```

This executes the example workflow described in [docs/workflow_example.md](docs/workflow_example.md).

## Public Data Sources

The examples in this repository can be trained on freely available datasets such as:

- [ChEMBL](https://www.ebi.ac.uk/chembl/)
- [BindingDB](https://www.bindingdb.org)
- [PDB](https://www.rcsb.org)

### Downloading example datasets

You can download the ESOL solubility dataset used in the GCN demo with:

```python
from data.dataset_download import download_esol_dataset

esol_df = download_esol_dataset(use_sample=True)  # set to False for the full dataset
```

Additional helpers allow fetching data directly from **ChEMBL** and **BindingDB**:

```python
from data.dataset_download import download_chembl_activity_data, download_bindingdb_dataset

chembl_df = download_chembl_activity_data("CHEMBL25", limit=50)
binding_db = download_bindingdb_dataset(use_sample=True)
```


## Contributing

Pull requests are welcome.  Feel free to open issues for bugs or feature requests.

## License

This project is licensed under the [MIT License](LICENSE).
