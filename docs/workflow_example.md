# Example End-to-End Workflow

This guide describes how to run the simple pipeline included with **Alpha Drug Discovery**.

```bash
python run.py pipeline --config config.yaml
```

The pipeline performs the following steps:

1. Downloads the ESOL dataset (sample only by default).
2. Trains a small variational autoencoder to generate molecules.
3. Generates a few candidate molecules and fits a toy ADMET model.
4. Produces a PDF report summarising the run.

Adjust parameters in `config.yaml` to customise the workflow.
