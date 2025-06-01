# ADMET Prediction Guide

This document explains how to train and use the ADMET prediction model included in **Alpha Drug Discovery**. The model estimates absorption, distribution, metabolism, excretion and toxicity properties for small molecules.

## Overview

The module `alpha_drug_discovery.admet_prediction` implements a simple feed-forward neural network. It can be trained with molecular descriptors or any numerical features you choose. The default output dimension is five, representing a set of ADMET endpoints.

Key features:

- **Flexible input**: provide any float arrays containing your descriptors.
- **Lightweight architecture**: small network with three dense layers.
- **Standalone usage**: train the model directly from your scripts or via the provided command-line runner.

## Installation

Make sure the project dependencies are installed:

```bash
pip install -r requirements.txt
```

The model requires **PyTorch**. If it is not available in your environment you can install the CPU-only version with:

```bash
pip install torch
```

## Training the Model

You need two arrays: `X` for the features and `y` for the target ADMET properties. Each row corresponds to a molecule. Example usage:

```python
from alpha_drug_discovery.admet_prediction import train_admet_model
import numpy as np

# 10 molecules, 8 descriptor values each
X = np.random.rand(10, 8).astype(np.float32)
# 5 ADMET properties per molecule
y = np.random.rand(10, 5).astype(np.float32)

model = train_admet_model(X, y, epochs=20, batch_size=16)
```

During training the function prints the loss for each epoch. The returned `model` is a `torch.nn.Module` that can be saved or used for prediction.

## Making Predictions

After training you can call the model on new feature arrays to obtain predicted properties:

```python
model.eval()
new_features = np.random.rand(3, 8).astype(np.float32)
with torch.no_grad():
    preds = model(torch.tensor(new_features))
print(preds)
```

## Using the Command Line Runner

The `run.py` script exposes an `admet_prediction` task. You can train the model interactively without writing a script:

```bash
python run.py
# When prompted, enter: admet_prediction
```

Inside `run_admet_prediction()` you should load or generate your data and pass it to `train_admet_model()`.

## Customising Hyperparameters

`train_admet_model` accepts several optional parameters:

- `epochs` – number of training epochs (default: `10`)
- `learning_rate` – Adam optimiser learning rate (default: `0.001`)
- `batch_size` – mini-batch size for the data loader (default: `32`)

Adjust these values according to your dataset size and convergence behaviour.

## Saving and Loading Models

You can persist trained models with `torch.save` and load them later with `torch.load`:

```python
import torch

torch.save(model.state_dict(), "admet_model.pt")

new_model = ADMETModel(input_dim=X.shape[1], output_dim=y.shape[1])
new_model.load_state_dict(torch.load("admet_model.pt"))
```

## Tips

- Standardise or normalise your input features for better convergence.
- If you only have some ADMET endpoints, adjust the output dimension when creating `ADMETModel` or by providing a `y` array with the desired number of columns.
- Larger models or additional layers can be implemented if higher accuracy is required.

## Conclusion

The ADMET prediction module offers a lightweight starting point for property prediction tasks. Integrate it with your own descriptor pipelines or expand the architecture to suit more advanced use cases.

