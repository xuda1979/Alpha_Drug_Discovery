# run.py

# Import heavy modules lazily inside functions to allow execution without
# optional dependencies such as PyTorch.
# The heavyweight model modules are loaded lazily in the individual helper
# functions to avoid mandatory PyTorch installation.
from data import dataset_download
import numpy as np
import pandas as pd
import argparse

def run_gan_drug_design(dataset: str = "esol", epochs: int = 1, batch_size: int = 32) -> None:
    """Example GAN training on a small dataset.

    If PyTorch is unavailable, falls back to a lightweight numpy demo so that the
    command can run in restricted environments.
    """
    df = dataset_download.load_dataset(dataset, use_sample=False, cache_dir="data")
    # Ensure we are using relevant columns for features, e.g., 'measured log solubility in mols per litre'
    # For this example, we'll stick to simple float selection, assuming the target CSV is well-formed for this.
    # A more robust approach would be to specify feature columns.
    X = df.select_dtypes(include=[np.number]).values # Select all numeric columns as features
    if X.shape[1] == 0:
        print("Warning: No numeric features found in ESOL dataset for GAN. Using random data.")
        X = np.random.rand(100, 10).astype(np.float32) # Fallback to random data if no numeric features
    else:
        # Normalize features for GAN training
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8) # Adding epsilon to avoid division by zero

    try:
        from models import gan_drug_design
        import torch  # noqa: F401
        gan_drug_design.train_gan(X, epochs=epochs, batch_size=min(batch_size, len(X)))
    except Exception as exc:  # pragma: no cover - fallback for missing deps
        print(f"PyTorch unavailable ({exc}). Running simplified numpy demo...")
        for epoch in range(epochs):
            noise = np.random.randn(min(batch_size, len(X)), X.shape[1])
            fake = np.tanh(noise).mean(axis=0)
            if epoch % max(1, epochs // 2) == 0:
                print(f"Epoch {epoch+1}/{epochs} - mean: {fake.mean():.4f}")
        print("Simplified GAN training complete.")

def run_rl_drug_design(epochs: int = 1) -> None:
    """Train a policy network in a dummy environment.

    Uses PyTorch if available, otherwise performs a very small numpy based
    simulation so the command completes without heavy dependencies.
    """

    class DummyEnv:
        def __init__(self, state_dim: int, action_dim: int, max_steps: int = 5):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.max_steps = max_steps

        def reset(self):
            return np.zeros(self.state_dim)

        def step(self, action: int):
            next_state = np.random.rand(self.state_dim)
            reward = np.random.rand()
            done = False
            return next_state, reward, done, {}

    env = DummyEnv(10, 3)
    try:
        from models import rl_drug_design
        import torch  # noqa: F401
        policy_network = rl_drug_design.PolicyNetwork(state_dim=10, action_dim=3)
        rl_drug_design.train_policy_gradient(env, policy_network, epochs=epochs)
    except Exception as exc:  # pragma: no cover - fallback for missing deps
        print(f"PyTorch unavailable ({exc}). Running simplified numpy demo...")
        state = env.reset()
        for epoch in range(epochs):
            action = np.random.choice(env.action_dim)
            state, reward, _, _ = env.step(action)
            if epoch % max(1, epochs // 2) == 0:
                print(f"Epoch {epoch+1}/{epochs} - action: {action}, reward: {reward:.3f}")
        print("Simplified RL training complete.")

def run_deep_docking(epochs: int = 1) -> None:
    """Train the simple deep docking CNN on random grids.

    Falls back to a numpy implementation if PyTorch is unavailable.
    """
    X = np.random.rand(8, 1, 8, 8, 8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    try:
        from models import deep_docking
        import torch  # noqa: F401
        deep_docking.train_docking_model(X, y, epochs=epochs)
    except Exception as exc:  # pragma: no cover - fallback for missing deps
        print(f"PyTorch unavailable ({exc}). Running simplified numpy demo...")
        weights = np.random.rand(np.prod(X.shape[1:]))
        for epoch in range(epochs):
            preds = X.reshape(len(X), -1).dot(weights)
            preds = 1 / (1 + np.exp(-preds))
            loss = ((preds - y) ** 2).mean()
            if epoch % max(1, epochs // 2) == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f}")
        print("Simplified docking training complete.")

def run_admet_prediction(epochs: int = 1) -> None:
    """Run ADMET model training on random data.

    Uses the PyTorch implementation when available. If torch cannot be imported
    the function performs a small linear regression with NumPy as a stand-in so
    the example can still execute.
    """
    try:
        df = pd.read_csv("data/chembl_aspirin_activity.csv")
        # Select features and target - this is highly dependent on the actual CSV structure
        # For example, using 'pchembl_value' as target if available and numeric
        # And other numeric columns as features. This is a placeholder.
        if 'pchembl_value' in df.columns:
            df = df.dropna(subset=['pchembl_value']) # Drop rows where target is NaN
            y_series = pd.to_numeric(df['pchembl_value'], errors='coerce')
            y_series = y_series.dropna()

            potential_feature_cols = [col for col in df.columns if col not in ['pchembl_value', 'molecule_chembl_id', 'target_chembl_id', 'document_chembl_id', 'assay_chembl_id', 'standard_relation', 'data_validity_comment', 'potential_duplicate', 'standard_type', 'standard_units', 'activity_comment'] and df[col].dtype in [np.number, 'float64', 'int64']]

            # Attempt to find at least a few numeric feature columns that are not mostly NaN
            feature_cols = []
            for col in potential_feature_cols:
                if df[col].notna().sum() > 0.5 * len(df): # More than 50% non-NaN values
                    feature_cols.append(col)
                if len(feature_cols) >= 5: # Take up to 5 such features for simplicity
                    break

            if not feature_cols: # If no good feature columns found
                print("Warning: Could not find suitable numeric feature columns in ChEMBL dataset. Using random data for ADMET.")
                X = np.random.rand(10, 8).astype(np.float32)
                y = np.random.rand(10, 5).astype(np.float32)
            else:
                X_df = df[feature_cols].fillna(0) # Fill NaNs with 0 for simplicity
                X = X_df.loc[y_series.index].values.astype(np.float32) # Align X with y_series
                y = y_series.values.reshape(-1, 1).astype(np.float32)

                if X.shape[0] == 0 or y.shape[0] == 0 or X.shape[0] != y.shape[0]:
                    print(f"Warning: Not enough data after processing ChEMBL dataset (X: {X.shape}, y: {y.shape}). Using random data for ADMET.")
                    X = np.random.rand(10, 8).astype(np.float32)
                    y = np.random.rand(10, 5).astype(np.float32)
                else:
                     # Normalize features
                    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        else:
            print("Warning: 'pchembl_value' not found in ChEMBL dataset. Using random data for ADMET.")
            X = np.random.rand(10, 8).astype(np.float32)
            y = np.random.rand(10, 5).astype(np.float32)

    except FileNotFoundError:
        print("Warning: data/chembl_aspirin_activity.csv not found. Using random data for ADMET.")
        X = np.random.rand(10, 8).astype(np.float32)
        y = np.random.rand(10, 5).astype(np.float32)
    except Exception as e: # Catch any other pandas processing error
        print(f"Error processing ChEMBL data: {e}. Using random data for ADMET.")
        X = np.random.rand(10, 8).astype(np.float32)
        y = np.random.rand(10, 5).astype(np.float32)


    try:
        from alpha_drug_discovery import admet_prediction # Original import was this
        import torch  # noqa: F401
        # Ensure y has the correct shape if it's 1D
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        # If admet_prediction.train_admet_model expects y to have multiple columns (e.g. 5 like the random data)
        # we might need to replicate y or adjust the model. For now, assume it can handle y.shape[1]
        if X.shape[0] > 0 : # Proceed only if there is data
             if y.shape[1] != 5 and X.shape[0] > 10 : # if y is not like the original random shape, and we have enough data, limit X for demo
                  # This part is tricky as the ADMET model might be expecting a certain number of output tasks (y.shape[1])
                  # For now, if y is a single column, we'll proceed. The model might need adjustment if it's hardcoded for 5 outputs.
                  print(f"Proceeding with ADMET training. X shape: {X.shape}, y shape: {y.shape}")
             admet_prediction.train_admet_model(X, y, epochs=epochs)
        else:
            print("Skipping ADMET training due to no data after processing.")

    except Exception as exc:  # pragma: no cover - fallback for missing deps
        print(f"PyTorch unavailable or other ADMET model error ({exc}). Running simplified numpy demo...")
        # Ensure X and y for numpy demo are of compatible shapes if previous loading failed
        if 'X' not in locals() or 'y' not in locals() or X.shape[0] == 0:
             X = np.random.rand(10, 8).astype(np.float32)
             y = np.random.rand(10, 5).astype(np.float32)
        elif len(y.shape) == 1: # Ensure y is 2D for dot product
            y = y.reshape(-1,1)

        # Adjust weights for numpy demo based on actual y shape
        weights_dim_1 = y.shape[1] if len(y.shape) > 1 else 1
        weights = np.random.rand(X.shape[1], weights_dim_1)

        for epoch in range(epochs):
            preds = X.dot(weights)
            loss = ((preds - y) ** 2).mean()
            grad = X.T.dot(preds - y) / len(X)
            weights -= 0.01 * grad
            if epoch % max(1, epochs // 2) == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f}")
        print("Simplified ADMET training complete.")

# Add similar functions for other new components...

def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Drug Discovery runner")
    parser.add_argument("task", help="Task to run")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    task = args.task
    if task == "gan_design":
        run_gan_drug_design()
    elif task == "rl_design":
        run_rl_drug_design()
    elif task == "deep_docking":
        run_deep_docking()
    elif task == "admet_prediction":
        run_admet_prediction()
    elif task == "pipeline":
        from alpha_drug_discovery import workflows
        workflows.basic_drug_discovery_pipeline(args.config)
    elif task == "full_demo":
        from alpha_drug_discovery import workflows
        workflows.full_feature_pipeline(args.config)
    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
