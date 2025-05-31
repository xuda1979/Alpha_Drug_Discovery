# tests/test_rl_drug_design.py

import unittest
import importlib
from unittest.mock import MagicMock

try:
    torch = importlib.import_module('torch')
    from models.rl_drug_design import train_policy_gradient, PolicyNetwork
except Exception as e:  # pragma: no cover - skip if deps missing
    torch = None
    train_policy_gradient = None
    PolicyNetwork = None
    IMPORT_ERROR = e

class TestRLDrugDesign(unittest.TestCase):
    def test_train_policy_gradient(self):
        if torch is None or train_policy_gradient is None:
            self.skipTest(f'Dependencies missing: {IMPORT_ERROR}')

        env = MagicMock()
        env.reset.return_value = [0.0] * 10
        env.step.return_value = ([0.0] * 10, 1.0, False, {})
        env.max_steps = 5

        policy_network = PolicyNetwork(state_dim=10, action_dim=3)
        trained_policy = train_policy_gradient(env, policy_network, epochs=10)
        self.assertIsNotNone(trained_policy)

if __name__ == '__main__':
    unittest.main()
